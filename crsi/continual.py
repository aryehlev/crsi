"""Source RSI and the paper's four continuation conditions (section 3.3).

    Cold Start            (H_0, none)    seed harness, no memory
    Harness Transfer      (H_S, none)    source-evolved harness, no memory
    Problem-Only Memory   (H_S, M_S^c)   source-evolved harness, root causes only
    Full SIM              (H_S, M_S)     source-evolved harness and the full record

Every condition keeps its memory setting through the target run, and every condition keeps the
ordinary history of its own run.
"""

from __future__ import annotations

from pathlib import Path

from .core import RunError, read_json
from .loop import RunConfig, optimize
from .memory import SIM

ARMS = ("cold_start", "harness_transfer", "problem_only", "full_sim")
_MODE = {"cold_start": "none", "harness_transfer": "none", "problem_only": "problems", "full_sim": "full"}


def run_source(bench, harness_dir: str | Path, proposer, selector, config: RunConfig) -> dict:
    """Source RSI: start from a harness and an empty memory, produce (H_S, M_S) in `config.out`."""
    return optimize(bench, harness_dir, proposer, selector, config, memory=SIM(), memory_mode="full", arm="source")


def continue_rsi(
    bench,
    arm: str,
    proposer,
    selector,
    config: RunConfig,
    source_dir: str | Path | None = None,
    seed_harness: str | Path | None = None,
) -> dict:
    """Target RSI on `bench` under one of the four conditions.

    `source_dir` is a finished run directory (a source run, or an earlier continuation).
    `seed_harness` is H_0 and is needed for cold_start only.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    source = Path(source_dir) if source_dir is not None else None
    source_name = None
    if source is not None:
        if not (source / "result.json").is_file():
            raise RunError(f"{source} is not a finished run (no result.json)")
        source_name = read_json(source / "result.json")["name"]

    if arm == "cold_start":
        if seed_harness is None:
            raise RunError("cold_start starts from the seed harness: pass seed_harness")
        return optimize(bench, seed_harness, proposer, selector, config, memory=None, memory_mode="none", arm=arm, source=source_name)

    if source is None:
        raise RunError(f"{arm} continues an earlier run: pass source_dir")
    memory = None
    if _MODE[arm] != "none":
        if not (source / "sim.json").is_file():
            raise RunError(f"{source} has no memory to carry (it was run without one)")
        domain = read_json(source / "result.json").get("domain", "")
        memory = SIM.load(source / "sim.json").carry(source_name, domain, config.name, problems_only=arm == "problem_only")
    return optimize(bench, source / "retained", proposer, selector, config, memory=memory, memory_mode=_MODE[arm], arm=arm, source=source_name)
