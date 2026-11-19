"""A deterministic benchmark that needs no model: each task kind is one string operation.

The seed harness returns the input unchanged and fails everything. A harness passes a kind once
it has a handler for it.
"""

from __future__ import annotations

from crsi import TaskResult, load_harness

OPS = {
    "strip": lambda s: s.strip(),
    "reverse": lambda s: s[::-1],
    "upper": lambda s: s.upper(),
    "lower": lambda s: s.lower(),
}
EXPR = {"strip": "text.strip()", "reverse": "text[::-1]", "upper": "text.upper()", "lower": "text.lower()"}
SEED = 'def run(task, llm):\n    return task["text"]\n'


def harness_with(*kinds: str) -> str:
    """Source of a harness that handles exactly `kinds`."""
    lines = ["def run(task, llm):", '    text = task["text"]']
    for kind in kinds:
        lines += [f'    if task["kind"] == "{kind}":', f"        return {EXPR[kind]}"]
    return "\n".join(lines + ["    return text"]) + "\n"


class FakePolicy:
    def __init__(self, name: str = "model-1") -> None:
        self.name = name

    def complete(self, system: str, user: str) -> str:
        return ""


class ToyBench:
    domain = "toy"
    interface = "harness.py must define run(task, llm) -> str; task has the keys 'id', 'kind' and 'text'"

    def __init__(self, policy, name: str = "toy-A", kinds=("strip", "reverse", "upper"), per_kind: int = 2) -> None:
        self.llm, self.policy, self.name = policy, policy.name, name
        self.kinds, self.per_kind = tuple(kinds), per_kind

    def tasks(self, split: str) -> list[str]:
        offset = 0 if split == "evolve" else self.per_kind
        return [f"{kind}-{offset + i}" for kind in self.kinds for i in range(self.per_kind)]

    def get(self, task_id: str) -> dict:
        kind, number = task_id.rsplit("-", 1)
        return {"id": task_id, "kind": kind, "text": f"  Ab{number}  "}

    def run(self, harness_dir, task_id: str, rollout: int) -> TaskResult:
        task = self.get(task_id)
        answer = load_harness(harness_dir).run(task, self.llm)
        expected = OPS[task["kind"]](task["text"])
        ok = answer == expected
        return TaskResult(task_id, rollout, float(ok), ok, trace=f"input {task['text']!r} returned {answer!r} expected {expected!r}")


def factory(policy, **kwargs) -> ToyBench:
    if isinstance(kwargs.get("kinds"), str):
        kwargs["kinds"] = tuple(kwargs["kinds"].split("+"))
    return ToyBench(policy, **kwargs)
