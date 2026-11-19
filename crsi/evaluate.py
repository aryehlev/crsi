"""Run a harness on one split of a benchmark."""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .core import EvalReport, TaskResult
from .llm import LLMError


def _run_trial(bench, harness_dir: Path, task_id: str, rollout: int, timeout: float | None) -> TaskResult:
    box: dict = {}

    def call() -> None:
        try:
            box["result"] = bench.run(harness_dir, task_id, rollout)
        except LLMError as exc:  # the model is unreachable: not the harness's fault
            box["fatal"] = exc
        except BaseException:  # noqa: BLE001 - a broken harness is a failed trial, not a crash
            box["error"] = traceback.format_exc(limit=8)

    if timeout is None:
        call()
    else:
        # A hung task cannot be killed from here. The thread is abandoned and the trial fails.
        thread = threading.Thread(target=call, daemon=True, name=f"crsi-{task_id}-{rollout}")
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            message = f"timed out after {timeout:g}s"
            return TaskResult(task_id, rollout, 0.0, False, trace=message, error=message)
    if "fatal" in box:
        raise box["fatal"]
    if "error" in box:
        return TaskResult(task_id, rollout, 0.0, False, trace=box["error"], error=box["error"].strip().splitlines()[-1])
    result = box.get("result")
    if not isinstance(result, TaskResult):
        message = f"benchmark returned {type(result).__name__}, not a TaskResult"
        return TaskResult(task_id, rollout, 0.0, False, trace=message, error=message)
    return result


def evaluate(
    bench,
    harness_dir: str | Path,
    split: str,
    rollouts: int = 5,
    workers: int = 8,
    timeout: float | None = 600.0,
) -> EvalReport:
    """Every task of `split`, `rollouts` times. Raises LLMError if the model cannot be reached."""
    harness_dir = Path(harness_dir)
    trials = [(task_id, rollout) for task_id in bench.tasks(split) for rollout in range(rollouts)]
    if not trials:
        raise ValueError(f"benchmark {getattr(bench, 'name', bench)!r} has no tasks in split {split!r}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda trial: _run_trial(bench, harness_dir, trial[0], trial[1], timeout), trials))
    return EvalReport(split, results)
