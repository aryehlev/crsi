"""The optimization loop.

    C_t     = P(H_t, F_t, D_t; M_t)              k candidate edits
    H_{t+1} = S(H_t, F_t, C_t, F_t^C; M_t)       at most one is kept
    M_{t+1} = U(M_t, e_t)                        the kept edit is recorded with its outcomes

The optimizer sees evolve-set results only. The held-out split is scored once before the first
round and once after the last, and those scores never reach the proposer or the selector.
"""

from __future__ import annotations

import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .core import MEMORY_MODES, Candidate, Context, EvalReport, ProposalError, RunError, read_json, write_json
from .evaluate import evaluate
from .harness import changed_files, copy_harness, diff_files, fingerprint, read_files, syntax_errors
from .llm import LLMError
from .memory import SIM, Attempt

RECORD_MODES = ("selected", "evaluated")
RESUME_KEYS = ("name", "candidates", "rollouts", "threshold", "record", "memory_mode", "arm", "source", "benchmark", "policy")


@dataclass
class RunConfig:
    name: str = "run"
    out: str = "runs/run"
    rounds: int = 10  # N in the paper
    candidates: int = 5  # k
    rollouts: int = 5  # per task, on both splits
    threshold: float = 0.5  # a task is solved when its mean score over rollouts reaches this
    record: str = "selected"  # or "evaluated": also log evaluated edits the selector rejected
    workers: int = 8  # tasks run in parallel
    timeout: float | None = 600.0  # seconds per task rollout
    verbose: bool = True


def optimize(
    bench,
    start_harness: str | Path,
    proposer,
    selector,
    config: RunConfig,
    memory: SIM | None = None,
    memory_mode: str = "full",
    arm: str = "source",
    source: str | None = None,
) -> dict:
    """Run (or resume) harness optimization in `config.out` and return the run's result."""
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
    if config.record not in RECORD_MODES:
        raise ValueError(f"record must be one of {RECORD_MODES}")
    if config.rounds < 0 or config.candidates < 1 or config.rollouts < 1:
        raise ValueError("rounds must be >= 0, candidates and rollouts >= 1")
    run = _Run(bench, proposer, selector, config, memory_mode, arm, source)
    return run.execute(Path(start_harness), memory)


class _Run:
    def __init__(self, bench, proposer, selector, config: RunConfig, memory_mode: str, arm: str, source: str | None):
        self.bench, self.proposer, self.selector, self.cfg = bench, proposer, selector, config
        self.mode, self.arm, self.source = memory_mode, arm, source
        self.out = Path(config.out)
        self.memory: SIM | None = None
        self.state: dict = {}

    # -- plumbing ---------------------------------------------------------------------------

    def log(self, message: str) -> None:
        if self.cfg.verbose:
            print(f"[{self.cfg.name}] {message}", file=sys.stderr, flush=True)

    def evaluate(self, harness_dir: Path, split: str) -> EvalReport:
        return evaluate(self.bench, harness_dir, split, self.cfg.rollouts, self.cfg.workers, self.cfg.timeout)

    def describe(self) -> dict:
        return {
            **{k: v for k, v in asdict(self.cfg).items() if k not in ("out", "verbose", "workers", "timeout")},
            "memory_mode": self.mode,
            "arm": self.arm,
            "source": self.source,
            "benchmark": self.bench.name,
            "domain": self.bench.domain,
            "policy": self.bench.policy,
        }

    def save_state(self) -> None:
        write_json(self.out / "state.json", self.state)
        copy_harness(self.out / self.state["retained"], self.out / "retained")
        if self.memory is not None:
            self.memory.save(self.out / "sim.json")

    # -- start and resume -------------------------------------------------------------------

    def execute(self, start_harness: Path, memory: SIM | None) -> dict:
        config_path, state_path = self.out / "config.json", self.out / "state.json"
        if self.out.exists() and any(self.out.iterdir()) and not config_path.exists():
            raise RunError(f"{self.out} is not empty and is not a crsi run directory")
        if state_path.exists():
            self._resume(read_json(config_path))
        else:
            self._start(start_harness, memory)
        if self.state["status"] == "done" and self.state["completed"] >= self.cfg.rounds:
            return read_json(self.out / "result.json")
        write_json(config_path, self.describe())
        for number in range(self.state["completed"] + 1, self.cfg.rounds + 1):
            self._round(number)
        return self._finish()

    def _start(self, start_harness: Path, memory: SIM | None) -> None:
        for stale in ("start", "rounds", "retained"):
            shutil.rmtree(self.out / stale, ignore_errors=True)
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(self.out / "config.json", self.describe())
        copy_harness(start_harness, self.out / "start" / "harness")
        if self.mode != "none":
            self.memory = memory if memory is not None else SIM()
            self.memory.problem_only = self.mode == "problems"
            self.memory.save(self.out / "start" / "sim.json")
        self.log(f"{self.arm} on {self.bench.name} with policy {self.bench.policy}; memory: {self.mode}")
        evolve = self.evaluate(self.out / "start" / "harness", "evolve")
        evolve.save(self.out / "start" / "evolve.json")
        heldout = self.evaluate(self.out / "start" / "harness", "heldout")
        heldout.save(self.out / "heldout_before.json")
        self.log(f"start: evolve {evolve.score:.2f}, held-out {heldout.score:.2f}")
        self.state = {
            "status": "running",
            "completed": 0,
            "retained": "start/harness",
            "evolve": "start/evolve.json",
            "sim": "start/sim.json" if self.mode != "none" else None,
        }
        self.save_state()

    def _resume(self, saved: dict) -> None:
        now = self.describe()
        changed = [k for k in RESUME_KEYS if saved.get(k) != now.get(k)]
        if changed:
            detail = ", ".join(f"{k}: {saved.get(k)!r} -> {now.get(k)!r}" for k in changed)
            raise RunError(f"{self.out} holds a run with different settings ({detail}). Use another --out.")
        self.state = read_json(self.out / "state.json")
        if self.state.get("sim"):
            self.memory = SIM.load(self.out / self.state["sim"])
        self.save_state()  # re-sync retained/ and sim.json with the last completed round
        if self.state["completed"] < self.cfg.rounds:
            self.log(f"resuming after round {self.state['completed']}")

    def _history(self) -> list[dict]:
        return [
            read_json(self.out / "rounds" / f"r{n:02d}" / "round.json")
            for n in range(1, self.state["completed"] + 1)
        ]

    # -- one round --------------------------------------------------------------------------

    def _round(self, number: int) -> None:
        rdir = self.out / "rounds" / f"r{number:02d}"
        shutil.rmtree(rdir, ignore_errors=True)  # left over from an interrupted round
        rdir.mkdir(parents=True)
        harness_dir = self.out / self.state["retained"]
        ctx = Context(
            run=self.cfg.name,
            round=number,
            rounds=self.cfg.rounds,
            candidates=self.cfg.candidates,
            benchmark=self.bench.name,
            domain=self.bench.domain,
            policy=self.bench.policy,
            interface=getattr(self.bench, "interface", ""),
            harness_dir=harness_dir,
            harness=read_files(harness_dir),
            evolve=EvalReport.load(self.out / self.state["evolve"]),
            history=self._history(),
            memory=self.memory,
            memory_mode=self.mode,
            threshold=self.cfg.threshold,
        )
        candidates = self._propose(ctx, rdir)
        self._static_screen(ctx, candidates)
        live = [c for c in candidates if c.status == "proposed"]
        if live:
            rejected = self.selector.screen(ctx, live) or {}
            for cand in live:
                if cand.index in rejected:
                    cand.status, cand.reason = "screened_out", f"selector: {rejected[cand.index]}"
        before = ctx.evolve.solved(self.cfg.threshold)
        for cand in candidates:
            if cand.status != "proposed":
                continue
            cand.report = self.evaluate(cand.harness_dir, "evolve")
            cand.report.save(cand.dir / "eval.json")
            after = cand.report.solved(self.cfg.threshold)
            cand.fixed_tasks, cand.regressed_tasks = sorted(after - before), sorted(before - after)
            cand.status = "evaluated"
        evaluated = [c for c in candidates if c.status == "evaluated"]
        choice, reason = self.selector.select(ctx, evaluated) if evaluated else (None, "no candidate was evaluated")
        chosen = next((c for c in evaluated if c.index == choice), None)

        problem_id = None
        if self.memory is not None:
            for cand in evaluated:
                if cand is chosen:
                    problem_id = self._record(ctx, cand, accepted=True)
                elif self.cfg.record == "evaluated":
                    self._record(ctx, cand, accepted=False)
            self.memory.save(rdir / "sim.json")

        write_json(rdir / "round.json", {
            "round": number,
            "evolve_before": ctx.evolve.score,
            "evolve_after": chosen.report.score if chosen else ctx.evolve.score,
            "selected": chosen.index if chosen else None,
            "selection_reason": reason,
            "problem": problem_id,
            "selected_diff": chosen.diff if chosen else "",
            "candidates": [self._summary(c) for c in candidates],
        })
        if chosen:
            rel = f"rounds/r{number:02d}/c{chosen.index}"
            self.state["retained"], self.state["evolve"] = f"{rel}/harness", f"{rel}/eval.json"
        if self.memory is not None:
            self.state["sim"] = f"rounds/r{number:02d}/sim.json"
        self.state["completed"] = number
        self.state["status"] = "running"
        self.save_state()
        counts = f"{len(evaluated)}/{len(candidates)} evaluated"
        if chosen:
            self.log(f"round {number}: {counts}; kept c{chosen.index}, evolve {ctx.evolve.score:.2f} -> {chosen.report.score:.2f}")
        else:
            self.log(f"round {number}: {counts}; nothing kept ({reason})")

    def _propose(self, ctx: Context, rdir: Path) -> list[Candidate]:
        def one(index: int) -> tuple[Candidate, LLMError | None]:
            cand = Candidate(index, rdir / f"c{index}")
            cand.dir.mkdir(parents=True)
            copy_harness(ctx.harness_dir, cand.harness_dir)
            try:
                cand.manifest = self.proposer.propose(ctx, index, cand.dir)
            except ProposalError as exc:
                cand.status, cand.reason = "failed", f"unusable proposal: {exc}"
                return cand, None
            except LLMError as exc:
                cand.status, cand.reason = "failed", f"model error: {exc}"
                return cand, exc
            except Exception as exc:  # noqa: BLE001 - one bad proposal must not end the run
                cand.status, cand.reason = "failed", f"proposer error: {type(exc).__name__}: {exc}"
                return cand, None
            files = read_files(cand.harness_dir)
            cand.diff = diff_files(ctx.harness, files)
            cand.files = changed_files(ctx.harness, files)
            (cand.dir / "edit.diff").write_text(cand.diff + ("\n" if cand.diff else ""), encoding="utf-8")
            known = set(ctx.evolve.task_ids())
            manifest = cand.manifest
            manifest.files = cand.files
            manifest.ignored_predictions = [p for p in manifest.predicted if p not in known]
            manifest.predicted = [p for p in manifest.predicted if p in known]
            write_json(cand.dir / "manifest.json", manifest.to_dict())
            return cand, None

        with ThreadPoolExecutor(max_workers=self.cfg.candidates) as pool:
            outcomes = list(pool.map(one, range(self.cfg.candidates)))
        errors = [error for _, error in outcomes if error is not None]
        if len(errors) == len(outcomes):
            raise errors[0]  # the optimizer model is unreachable: stop, the run can be resumed
        return [cand for cand, _ in outcomes]

    def _static_screen(self, ctx: Context, candidates: list[Candidate]) -> None:
        """Checks that need no model: empty, duplicate, syntax error, bad citation, benchmark hook."""
        seen: dict[str, int] = {}
        validate = getattr(self.bench, "validate", None)
        for cand in candidates:
            if cand.status != "proposed":
                continue
            files = read_files(cand.harness_dir)
            reason = ""
            if not cand.diff:
                reason = "empty: the proposal does not change the harness"
            elif fingerprint(files) in seen:
                reason = f"duplicate of c{seen[fingerprint(files)]}"
            elif syntax_errors(files):
                reason = "syntax error: " + "; ".join(syntax_errors(files))
            elif self.memory is not None and self.memory.get(cand.manifest.problem_id) is None and not cand.manifest.root_cause:
                cited = cand.manifest.problem_id
                reason = (f"the manifest cites {cited!r}, which is not in the memory, " if cited else "the manifest ") + "names no root cause"
            elif validate is not None:
                reason = validate(cand.harness_dir) or ""
                reason = f"benchmark check: {reason}" if reason else ""
            if reason:
                cand.status, cand.reason = "screened_out", reason
            else:
                seen[fingerprint(files)] = cand.index

    def _record(self, ctx: Context, cand: Candidate, accepted: bool) -> str:
        """U(M, e): file the edit under its root cause, with what it was predicted to do and what it did."""
        manifest = cand.manifest
        fixed = [task for task in manifest.predicted if task in cand.fixed_tasks]
        attempt = Attempt(
            round=ctx.round,
            sample=cand.index,
            accepted=accepted,
            what=manifest.what,
            why=manifest.why,
            files=list(cand.files),
            predicted=list(manifest.predicted),
            fixed=fixed,
            missed=[task for task in manifest.predicted if task not in fixed],
            passed=cand.report.passed,
            total=cand.report.total,
            domain=ctx.domain,
            benchmark=ctx.benchmark,
            policy=ctx.policy,
            run=ctx.run,
        )
        return self.memory.record(attempt, manifest.problem_id, manifest.root_cause).id

    @staticmethod
    def _summary(cand: Candidate) -> dict:
        added, removed = cand.diff_stat()
        return {
            "index": cand.index,
            "status": cand.status,
            "reason": cand.reason,
            "manifest": cand.manifest.to_dict() if cand.manifest else None,
            "score": cand.report.score if cand.report else None,
            "passed": cand.report.passed if cand.report else None,
            "total": cand.report.total if cand.report else None,
            "fixed_tasks": cand.fixed_tasks,
            "regressed_tasks": cand.regressed_tasks,
            "lines_added": added,
            "lines_removed": removed,
        }

    # -- finish -----------------------------------------------------------------------------

    def _finish(self) -> dict:
        heldout = self.evaluate(self.out / self.state["retained"], "heldout")
        heldout.save(self.out / "heldout_after.json")
        before = EvalReport.load(self.out / "heldout_before.json")
        history = self._history()
        result = {
            **self.describe(),
            "rounds": self.state["completed"],
            "accepted": sum(1 for entry in history if entry["selected"] is not None),
            "evolve_before": EvalReport.load(self.out / "start" / "evolve.json").score,
            "evolve_after": EvalReport.load(self.out / self.state["evolve"]).score,
            "heldout_before": before.score,
            "heldout_after": heldout.score,
            "improvement": heldout.score - before.score,
            "problems": len(self.memory) if self.memory is not None else 0,
            "attempts": self.memory.attempt_count() if self.memory is not None else 0,
        }
        write_json(self.out / "result.json", result)
        self.state["status"] = "done"
        self.save_state()
        self.log(f"done: held-out {before.score:.2f} -> {heldout.score:.2f} ({result['improvement']:+.2f})")
        return result
