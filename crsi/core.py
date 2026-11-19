"""Shared data types: task results, evaluation reports, manifests, candidates, the optimizer's context."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory import SIM

MEMORY_MODES = ("full", "problems", "none")


class ProposalError(Exception):
    """A proposal that cannot be used: unparseable reply, missing manifest, unsafe path."""


class RunError(Exception):
    """A run cannot start or resume as asked."""


def write_json(path: str | Path, obj: Any) -> None:
    """Write JSON through a temporary file so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class TaskResult:
    """One rollout of one task. `trace` is what the optimizer reads about it."""

    task_id: str
    rollout: int
    score: float
    passed: bool
    trace: str = ""
    error: str | None = None


@dataclass
class EvalReport:
    """Every rollout of every task of one split, for one harness."""

    split: str
    results: list[TaskResult] = field(default_factory=list)

    def task_ids(self) -> list[str]:
        return list(dict.fromkeys(r.task_id for r in self.results))

    def for_task(self, task_id: str) -> list[TaskResult]:
        return [r for r in self.results if r.task_id == task_id]

    def task_scores(self) -> dict[str, float]:
        """Mean score over rollouts, per task, in [0, 1]."""
        sums: dict[str, list[float]] = {}
        for r in self.results:
            sums.setdefault(r.task_id, []).append(r.score)
        return {task: sum(scores) / len(scores) for task, scores in sums.items()}

    @property
    def score(self) -> float:
        """Mean of the per-task means, in points (0 to 100)."""
        scores = self.task_scores()
        return 100.0 * sum(scores.values()) / len(scores) if scores else 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    def solved(self, threshold: float = 0.5) -> set[str]:
        return {task for task, score in self.task_scores().items() if score >= threshold}

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "score": self.score,
            "passed": self.passed,
            "total": self.total,
            "results": [asdict(r) for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalReport":
        return cls(data["split"], [TaskResult(**r) for r in data["results"]])

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        return cls.from_dict(read_json(path))


@dataclass
class Manifest:
    """The change manifest a proposer commits to before its edit is evaluated."""

    what: str
    why: str = ""
    problem_id: str | None = None
    root_cause: str = ""
    predicted: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    ignored_predictions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "Manifest":
        if not isinstance(data, dict):
            raise ProposalError("the manifest is not a JSON object")
        what = _text(data.get("what"))
        if not what:
            raise ProposalError("the manifest has no 'what'")
        problem_id = _text(data.get("problem_id"))
        if problem_id.lower() in ("", "null", "none", "new"):
            problem_id = ""
        predicted = data.get("predicted") or []
        if isinstance(predicted, str):
            predicted = [p for p in predicted.replace(",", " ").split() if p]
        if not isinstance(predicted, list):
            predicted = []
        return cls(
            what=what,
            why=_text(data.get("why")),
            problem_id=problem_id or None,
            root_cause=_text(data.get("root_cause")),
            predicted=list(dict.fromkeys(str(p).strip() for p in predicted if str(p).strip())),
        )

    def to_dict(self) -> dict:
        out = {
            "problem_id": self.problem_id,
            "root_cause": self.root_cause,
            "what": self.what,
            "why": self.why,
            "predicted": self.predicted,
            "files": self.files,
        }
        if self.ignored_predictions:
            out["ignored_predictions"] = self.ignored_predictions
        return out


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


@dataclass
class Candidate:
    """One proposal in one round, from proposal to evaluation."""

    index: int
    dir: Path
    manifest: Manifest | None = None
    status: str = "proposed"  # proposed | failed | screened_out | evaluated
    reason: str = ""
    diff: str = ""
    files: list[str] = field(default_factory=list)
    report: EvalReport | None = None
    fixed_tasks: list[str] = field(default_factory=list)  # unsolved before, solved now
    regressed_tasks: list[str] = field(default_factory=list)  # solved before, unsolved now

    @property
    def label(self) -> str:
        return f"c{self.index}"

    @property
    def harness_dir(self) -> Path:
        return self.dir / "harness"

    def diff_stat(self) -> tuple[int, int]:
        added = removed = 0
        for line in self.diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return added, removed


@dataclass
class Context:
    """Everything the optimizer may look at in one round. It holds evolve-set results only:
    held-out scores never enter it."""

    run: str
    round: int
    rounds: int
    candidates: int
    benchmark: str
    domain: str
    policy: str
    interface: str
    harness_dir: Path
    harness: dict[str, str]
    evolve: EvalReport
    history: list[dict]
    memory: "SIM | None"
    memory_mode: str = "full"
    threshold: float = 0.5
