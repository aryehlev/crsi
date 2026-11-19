"""Self-Improvement Memory (SIM): section 3.2 and Appendix D of the paper.

A memory is a list of problems. A problem is a root-cause hypothesis and the attempts made
against it, in order. An attempt is an edit with its rationale, the tasks it was predicted to
fix, the tasks it did fix, and where that was observed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from .core import read_json, write_json


def _norm(text: str) -> str:
    """Root causes are the same when their text is, ignoring case and whitespace."""
    return " ".join(text.lower().split())


@dataclass
class Attempt:
    round: int
    sample: int
    accepted: bool
    what: str
    files: list[str] = field(default_factory=list)
    predicted: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    passed: int = 0
    total: int = 0
    domain: str = ""
    backfilled: bool = False  # in the paper's schema, undefined there; always False here
    # Not in the paper's schema: the rest of the provenance its text describes, and the rationale.
    why: str = ""
    benchmark: str = ""
    policy: str = ""
    run: str = ""

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "sample": self.sample,
            "accepted": self.accepted,
            "backfilled": self.backfilled,
            "what": self.what,
            "files": list(self.files),
            "predicted": list(self.predicted),
            "fixed": list(self.fixed),
            "missed": list(self.missed),
            "passed": self.passed,
            "total": self.total,
            "domain": self.domain,
            "why": self.why,
            "benchmark": self.benchmark,
            "policy": self.policy,
            "run": self.run,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Attempt":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass
class Problem:
    id: str
    name: str  # the root-cause hypothesis
    opened_round: int
    attempts: list[Attempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "opened_round": self.opened_round,
            "attempts": [a.to_dict() for a in self.attempts],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Problem":
        return cls(
            id=data["id"],
            name=data["name"],
            opened_round=data.get("opened_round", 0),
            attempts=[Attempt.from_dict(a) for a in data.get("attempts", [])],
        )


class SIM:
    """The memory M. `record` is the paper's update U(M, e); `render` is what the optimizer reads."""

    def __init__(self) -> None:
        self.next_id = 1
        self.carried_from: dict | None = None
        self.aliases: dict[str, str] = {}
        self.problems: list[Problem] = []
        # Problem-Only Memory: root causes are kept, attempts are never stored.
        self.problem_only = False

    def __len__(self) -> int:
        return len(self.problems)

    def attempt_count(self) -> int:
        return sum(len(p.attempts) for p in self.problems)

    # -- lookup -----------------------------------------------------------------------------

    def resolve(self, problem_id: str) -> str:
        """Follow aliases left by merges."""
        seen = set()
        while problem_id in self.aliases and problem_id not in seen:
            seen.add(problem_id)
            problem_id = self.aliases[problem_id]
        return problem_id

    def get(self, problem_id: str | None) -> Problem | None:
        if not problem_id:
            return None
        target = self.resolve(problem_id.strip())
        for problem in self.problems:
            if problem.id == target:
                return problem
        return None

    def find(self, root_cause: str) -> Problem | None:
        key = _norm(root_cause)
        for problem in self.problems:
            if _norm(problem.name) == key:
                return problem
        return None

    # -- update -----------------------------------------------------------------------------

    def record(self, attempt: Attempt, problem_id: str | None = None, root_cause: str = "") -> Problem:
        """Append `attempt` to the cited problem, or open a new problem named `root_cause`.

        The proposer decides the grouping by citing `problem_id`. A new problem whose text is
        identical to an existing one is merged into it, leaving an alias.
        """
        problem = self.get(problem_id)
        if problem is None:
            name = " ".join(root_cause.split())
            if not name:
                raise ValueError(f"no problem {problem_id!r} in the memory and no root cause given")
            problem = Problem(id=f"P{self.next_id}", name=name, opened_round=attempt.round)
            self.next_id += 1
            self.problems.append(problem)
        if not self.problem_only:
            problem.attempts.append(attempt)
        self.merge_identical()
        return self.get(problem.id)  # type: ignore[return-value]

    def merge_identical(self) -> dict[str, str]:
        """Merge problems with identical root-cause text into the earliest one, keeping every
        attempt. Returns the aliases this call created."""
        first: dict[str, Problem] = {}
        kept: list[Problem] = []
        created: dict[str, str] = {}
        for problem in self.problems:
            key = _norm(problem.name)
            if key in first:
                target = first[key]
                target.attempts.extend(problem.attempts)
                created[problem.id] = target.id
            else:
                first[key] = problem
                kept.append(problem)
        if created:
            self.problems = kept
            for alias, target_id in list(self.aliases.items()):
                if target_id in created:
                    self.aliases[alias] = created[target_id]
            self.aliases.update(created)
        return created

    # -- continuation -----------------------------------------------------------------------

    def carry(self, source_run: str, domain: str, into: str, problems_only: bool = False) -> "SIM":
        """A copy to continue from in another run. With `problems_only`, the root causes are
        carried and their attempt histories are dropped."""
        new = copy.deepcopy(self)
        new.carried_from = {"arm": source_run, "domain": domain, "into": into}
        new.problem_only = problems_only
        if problems_only:
            for problem in new.problems:
                problem.attempts = []
        return new

    # -- storage ----------------------------------------------------------------------------

    def to_dict(self) -> dict:
        out: dict = {
            "next_id": self.next_id,
            "carried_from": self.carried_from,
            "aliases": dict(self.aliases),
            "problems": [p.to_dict() for p in self.problems],
        }
        if self.problem_only:
            out["problem_only"] = True
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "SIM":
        sim = cls()
        sim.problems = [Problem.from_dict(p) for p in data.get("problems", [])]
        sim.next_id = data.get("next_id", len(sim.problems) + 1)
        sim.carried_from = data.get("carried_from")
        sim.aliases = dict(data.get("aliases") or {})
        sim.problem_only = bool(data.get("problem_only", False))
        return sim

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "SIM":
        return cls.from_dict(read_json(path))

    # -- what the optimizer reads -----------------------------------------------------------

    def render(self) -> str:
        """The whole memory as text. Nothing is retrieved, ranked or truncated."""
        lines: list[str] = []
        current = None
        if self.carried_from:
            cf = self.carried_from
            current = cf.get("into")
            lines.append(
                f"Carried from run {cf.get('arm')!r} (domain: {cf.get('domain')}) into run {current!r}."
            )
        if self.problem_only:
            lines.append("Root causes only: the edits tried against them and their outcomes are not kept.")
        if not self.problems:
            lines.append("The memory is empty: no root cause has been recorded yet.")
            return "\n".join(lines)
        for problem in self.problems:
            if lines:
                lines.append("")
            lines.append(f"## {problem.id}: {problem.name}")
            if self.problem_only:
                lines.append(f"first diagnosed in round {problem.opened_round}")
                continue
            count = f"{len(problem.attempts)} attempt(s)" if problem.attempts else "no attempts recorded"
            lines.append(f"first diagnosed in round {problem.opened_round}; {count}")
            for number, attempt in enumerate(problem.attempts, 1):
                lines.extend(_render_attempt(number, attempt, current))
        merged = [f"{alias} is now {target}" for alias, target in sorted(self.aliases.items())]
        if merged:
            lines.append("")
            lines.append("Merged root causes: " + "; ".join(merged) + ".")
        return "\n".join(lines)


def _ids(label: str, ids: list[str]) -> str:
    return f"{label} ({len(ids)}): {', '.join(sorted(ids))}" if ids else f"{label} (0)"


def _render_attempt(number: int, a: Attempt, current_run: str | None) -> list[str]:
    carried = f" (carried from run {a.run!r})" if current_run and a.run and a.run != current_run else ""
    verdict = "accepted" if a.accepted else "evaluated, not accepted"
    if a.backfilled:
        verdict += ", backfilled"
    where = f"domain={a.domain}, benchmark={a.benchmark}, policy={a.policy}"
    lines = [f"- attempt {number}: round {a.round}, sample {a.sample}{carried}; {verdict}; {where}"]
    lines.append(f"  what: {a.what}")
    if a.why:
        lines.append(f"  why: {a.why}")
    if a.files:
        lines.append(f"  files: {', '.join(a.files)}")
    lines.append("  " + _ids("predicted to fix", a.predicted))
    lines.append(
        f"  observed: {_ids('fixed', a.fixed)}; {_ids('missed', a.missed)}; "
        f"evolve trials passed {a.passed}/{a.total}"
    )
    return lines
