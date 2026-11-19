"""Command line: crsi run | continue | report | show | paired."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .benchmarks import load_benchmark
from .continual import ARMS, continue_rsi, run_source
from .core import RunError, read_json
from .llm import LLMError, make_llm
from .loop import RECORD_MODES, RunConfig
from .memory import SIM
from .proposer import CommandProposer, LLMProposer
from .selector import GreedySelector, LLMSelector
from .stats import paired


def _value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(text.lower(), text)


def _bench_args(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--bench-arg expects key=value, got {pair!r}")
        out[key] = _value(value)
    return out


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    defaults = RunConfig()
    parser.add_argument("--bench", required=True, help="arithmetic, strings, or module:factory")
    parser.add_argument("--bench-arg", action="append", default=[], metavar="KEY=VALUE", help="passed to the benchmark factory; repeatable")
    parser.add_argument("--out", required=True, help="run directory; repeating a command resumes it")
    parser.add_argument("--name", help="run name recorded in the memory (default: the directory name)")
    parser.add_argument("--policy", required=True, metavar="PROVIDER:MODEL", help="the fixed model that solves tasks")
    parser.add_argument("--optimizer", metavar="PROVIDER:MODEL", help="the model that proposes and selects harness edits")
    parser.add_argument("--proposer-model", metavar="PROVIDER:MODEL", help="overrides --optimizer for proposing")
    parser.add_argument("--selector-model", metavar="PROVIDER:MODEL", help="overrides --optimizer for selecting")
    parser.add_argument("--proposer-cmd", metavar="COMMAND", help="an agent CLI that proposes edits instead of a single model call")
    parser.add_argument("--proposer-timeout", type=float, default=1800.0, help="seconds allowed per --proposer-cmd call")
    parser.add_argument("--greedy", action="store_true", help="select by evolve score alone, without a selector model")
    parser.add_argument("--rounds", type=int, default=defaults.rounds)
    parser.add_argument("--candidates", type=int, default=defaults.candidates, help="proposals per round")
    parser.add_argument("--rollouts", type=int, default=defaults.rollouts, help="runs per task")
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help="mean score at which a task counts as solved")
    parser.add_argument("--record", choices=RECORD_MODES, default=defaults.record, help="'evaluated' also logs rejected edits that were evaluated")
    parser.add_argument("--workers", type=int, default=defaults.workers, help="tasks run in parallel")
    parser.add_argument("--timeout", type=float, default=defaults.timeout, help="seconds per task run; 0 for no limit")
    parser.add_argument("--max-tokens", type=int, help="output limit per model call")
    parser.add_argument("--quiet", action="store_true")


def _setup(args):
    options = {"max_tokens": args.max_tokens} if args.max_tokens else {}
    if args.proposer_cmd:
        proposer = CommandProposer(args.proposer_cmd, timeout=args.proposer_timeout)
    else:
        spec = args.proposer_model or args.optimizer
        if not spec:
            raise SystemExit("give --optimizer (or --proposer-model, or --proposer-cmd)")
        proposer = LLMProposer(make_llm(spec, **options))
    if args.greedy:
        selector = GreedySelector()
    else:
        spec = args.selector_model or args.optimizer
        if not spec:
            raise SystemExit("give --optimizer or --selector-model, or use --greedy to select by score")
        selector = LLMSelector(make_llm(spec, **options))
    bench = load_benchmark(args.bench, make_llm(args.policy, **options), **_bench_args(args.bench_arg))
    config = RunConfig(
        name=args.name or Path(args.out).resolve().name,
        out=args.out,
        rounds=args.rounds,
        candidates=args.candidates,
        rollouts=args.rollouts,
        threshold=args.threshold,
        record=args.record,
        workers=args.workers,
        timeout=args.timeout or None,
        verbose=not args.quiet,
    )
    return bench, proposer, selector, config


def _print_result(result: dict) -> None:
    print(
        f"{result['name']}: held-out {result['heldout_before']:.2f} -> {result['heldout_after']:.2f} "
        f"({result['improvement']:+.2f}); {result['accepted']} of {result['rounds']} rounds kept an edit"
    )


def cmd_run(args) -> int:
    bench, proposer, selector, config = _setup(args)
    _print_result(run_source(bench, args.harness, proposer, selector, config))
    return 0


def cmd_continue(args) -> int:
    bench, proposer, selector, config = _setup(args)
    _print_result(continue_rsi(bench, args.arm, proposer, selector, config, source_dir=args.source, seed_harness=args.seed_harness))
    return 0


def _row(run_dir: Path) -> list[str]:
    def num(value) -> str:
        return "-" if value is None else f"{value:.2f}"

    if (run_dir / "result.json").is_file():
        r = read_json(run_dir / "result.json")
        return [
            r["name"], r["arm"], r["benchmark"], r["policy"], f"{r['accepted']}/{r['rounds']}",
            num(r["evolve_before"]), num(r["evolve_after"]), num(r["heldout_before"]), num(r["heldout_after"]),
            f"{r['improvement']:+.2f}",
        ]
    if (run_dir / "config.json").is_file():
        c = read_json(run_dir / "config.json")
        done = read_json(run_dir / "state.json")["completed"] if (run_dir / "state.json").is_file() else 0
        return [c["name"], c["arm"], c["benchmark"], c["policy"], f"unfinished: {done}/{c['rounds']} rounds", "-", "-", "-", "-", "-"]
    raise SystemExit(f"{run_dir} is not a crsi run directory")


def cmd_report(args) -> int:
    header = ["run", "arm", "benchmark", "policy", "kept", "evolve0", "evolve", "heldout0", "heldout", "gain"]
    rows = [header] + [_row(Path(run)) for run in args.runs]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    return 0


def cmd_show(args) -> int:
    run_dir = Path(args.run)
    path = run_dir / "rounds" / f"r{args.round:02d}" / "sim.json" if args.round else run_dir / "sim.json"
    if run_dir.is_file():
        path = run_dir
    if not path.is_file():
        if (run_dir / "config.json").is_file() and read_json(run_dir / "config.json").get("memory_mode") == "none":
            print("This run has no memory (cold_start and harness_transfer run without one).")
            return 0
        raise SystemExit(f"no memory found at {path}")
    memory = SIM.load(path)
    print(json.dumps(memory.to_dict(), indent=2, ensure_ascii=False) if args.json else memory.render())
    return 0


def _scores(items: list[str]) -> list[float]:
    scores = []
    for item in items:
        try:
            scores.append(float(item))
        except ValueError:
            result = Path(item) / "result.json"
            if not result.is_file():
                raise SystemExit(f"{item} is neither a number nor a finished run directory") from None
            scores.append(float(read_json(result)["heldout_after"]))
    return scores


def cmd_paired(args) -> int:
    treatment, control = _scores(args.treatment), _scores(args.control)
    if len(treatment) != len(control):
        raise SystemExit(f"{len(treatment)} treatment scores but {len(control)} control scores: they are paired by position")
    try:
        print(paired(treatment, control, level=args.level).describe())
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crsi", description="Self-Improvement Memory for continual harness optimization (unofficial).")
    parser.add_argument("--version", action="version", version=f"crsi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="source RSI: evolve a harness on a benchmark, building a memory")
    _add_run_options(p)
    p.add_argument("--harness", required=True, help="directory of the starting harness")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("continue", help="continue optimization on another benchmark or model")
    _add_run_options(p)
    p.add_argument("--from", dest="source", help="a finished run directory to continue from")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed-harness", help="the seed harness; needed for cold_start")
    p.set_defaults(func=cmd_continue)

    p = sub.add_parser("report", help="compare runs: evolve and held-out scores before and after")
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("show", help="print a run's memory the way the optimizer reads it")
    p.add_argument("run", help="a run directory or a sim.json file")
    p.add_argument("--round", type=int, help="the memory as it was after this round")
    p.add_argument("--json", action="store_true", help="print the stored JSON instead")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("paired", help="paired t-interval and exact signed-rank test on final held-out scores")
    p.add_argument("--treatment", nargs="+", required=True, metavar="RUN_OR_SCORE", help="e.g. the full_sim runs")
    p.add_argument("--control", nargs="+", required=True, metavar="RUN_OR_SCORE", help="e.g. the harness_transfer runs, in the same order")
    p.add_argument("--level", type=float, default=0.95, help="confidence level of the interval")
    p.set_defaults(func=cmd_paired)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RunError, LLMError, ValueError, FileNotFoundError) as exc:
        print(f"crsi: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("crsi: interrupted; repeat the command to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
