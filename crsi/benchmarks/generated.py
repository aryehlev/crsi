"""Two small generated benchmarks for trying the loop: `arithmetic` and `strings`.

They are toys. Tasks are generated from a seed, the answer is graded by exact match, and the
seed harness in examples/ fails many of them for reasons a harness can fix (answer format,
letter-level and multi-step work the model gets wrong unaided). They share one harness interface,
so a harness evolved on one can be continued on the other.
"""

from __future__ import annotations

import random
import string
import traceback
from pathlib import Path

from ..core import TaskResult
from ..harness import load_harness
from ..llm import LLMError

INTERFACE = (
    "harness.py must define run(task, llm) -> str. `task` is a dict with the keys 'id' and "
    "'prompt'. `llm.complete(system, user) -> str` calls the fixed policy model, at most "
    "{max_calls} times per task. The returned string is the answer: it is compared with the "
    "expected answer exactly, after stripping surrounding whitespace. {answer_format} The "
    "harness may use the Python standard library but not the network or the file system. "
    "Other modules of the harness are imported relatively (from . import helpers)."
)


class _Recorder:
    """The policy model as the harness sees it: counts and records calls for the trace."""

    def __init__(self, llm, max_calls: int) -> None:
        self._llm, self._max = llm, max_calls
        self.name = getattr(llm, "name", "policy")
        self.calls: list[tuple[str, str, str]] = []
        self.fatal: LLMError | None = None

    def complete(self, system: str, user: str) -> str:
        if len(self.calls) >= self._max:
            raise RuntimeError(f"call budget exceeded: at most {self._max} model calls per task")
        try:
            reply = self._llm.complete(system, user)
        except LLMError as exc:
            self.fatal = exc  # re-raised by the benchmark even if the harness swallows it
            raise
        self.calls.append((str(system), str(user), str(reply)))
        return reply


def _short(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else f"{text[:limit]} [... {len(text) - limit} more characters]"


class GeneratedBench:
    def __init__(self, policy, name: str, domain: str, answer_format: str, tasks: list[tuple[str, str, str]], max_calls: int = 8):
        """`tasks` is a list of (id, prompt, answer); the first half is the evolve split."""
        self.llm = policy
        self.policy = getattr(policy, "name", str(policy))
        self.name, self.domain, self.max_calls = name, domain, max_calls
        self.interface = INTERFACE.format(max_calls=max_calls, answer_format=answer_format)
        half = len(tasks) // 2
        self._splits = {"evolve": [t[0] for t in tasks[:half]], "heldout": [t[0] for t in tasks[half:]]}
        self._prompt = {t[0]: t[1] for t in tasks}
        self._answer = {t[0]: t[2] for t in tasks}

    def tasks(self, split: str) -> list[str]:
        return list(self._splits[split])

    def get(self, task_id: str) -> dict:
        """The task as the harness sees it. The answer is not in it."""
        return {"id": task_id, "prompt": self._prompt[task_id]}

    def grade(self, task_id: str, answer) -> bool:
        return isinstance(answer, str) and answer.strip() == self._answer[task_id]

    def validate(self, harness_dir: Path) -> str | None:
        try:
            module = load_harness(harness_dir)
        except Exception as exc:  # noqa: BLE001
            return f"harness.py does not import: {type(exc).__name__}: {exc}"
        return None if callable(getattr(module, "run", None)) else "harness.py does not define run(task, llm)"

    def run(self, harness_dir: Path, task_id: str, rollout: int) -> TaskResult:
        recorder = _Recorder(self.llm, self.max_calls)
        lines = [f"task: {self._prompt[task_id]}"]
        answer, failure = None, None
        try:
            answer = load_harness(harness_dir).run(self.get(task_id), recorder)
        except Exception:  # noqa: BLE001 - the harness is model-written code
            failure = traceback.format_exc(limit=6)
        if recorder.fatal is not None:
            raise recorder.fatal
        for number, (system, user, reply) in enumerate(recorder.calls, 1):
            lines.append(f"--- model call {number} ---")
            lines.append(f"system: {_short(system)}")
            lines.append(f"user: {_short(user)}")
            lines.append(f"reply: {_short(reply)}")
        if not recorder.calls:
            lines.append("(the harness made no model call)")
        if failure:
            lines.append(f"harness raised:\n{failure.rstrip()}")
        elif isinstance(answer, str):
            lines.append(f"harness returned: {_short(answer)!r}")
        else:
            lines.append(f"harness returned {type(answer).__name__}, not a string: {_short(repr(answer))}")
        ok = failure is None and self.grade(task_id, answer)
        lines.append(f"expected: {self._answer[task_id]!r}")
        lines.append("result: " + ("pass" if ok else "fail"))
        return TaskResult(task_id, rollout, float(ok), ok, trace="\n".join(lines), error=failure.strip().splitlines()[-1] if failure else None)


def _build(kinds: dict, n: int, seed: int, salt: str) -> list[tuple[str, str, str]]:
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = random.Random(f"{salt}-{seed}")
    names = list(kinds)
    counter = dict.fromkeys(names, 0)
    tasks = []
    for i in range(n):
        kind = names[i % len(names)]
        prompt, answer = kinds[kind](rng)
        tasks.append((f"{kind}-{counter[kind]}", prompt, str(answer)))
        counter[kind] += 1
    return tasks


# -- arithmetic ------------------------------------------------------------------------------

def _chain(rng: random.Random) -> tuple[str, int]:
    a, b, c, d, e = (rng.randint(12, 97) for _ in range(5))
    f = rng.randint(3, 9)
    expr, value = rng.choice([
        (f"({a} + {b}) * {f} - {c} + {d} * {e}", (a + b) * f - c + d * e),
        (f"{a} * {b} - ({c} + {d}) * {f} + {e}", a * b - (c + d) * f + e),
        (f"{a} + {b} * {c} - {d} * {f} + {e}", a + b * c - d * f + e),
    ])
    return f"Compute the value of this expression: {expr}", value


def _word(rng: random.Random) -> tuple[str, int]:
    a, b, d = rng.randint(14, 48), rng.randint(12, 36), rng.randint(3, 9)
    pick = rng.randrange(4)
    if pick == 0:
        c = (a * b) % d + d * rng.randint(1, 4)
        return (
            f"A warehouse receives {a} crates, each holding {b} boxes. {c} boxes are damaged and thrown "
            f"away. The rest are split equally among {d} shops. How many boxes does each shop get?",
            (a * b - c) // d,
        )
    if pick == 1:
        c = rng.randint(40, 300)
        return (
            f"A train travels {a * 10} km on each of {b} days and {c} km on one last day. "
            "How many km does it travel in total?",
            a * 10 * b + c,
        )
    if pick == 2:
        c = rng.randint(50, a * b - 10)
        return (
            f"{a} students each pay {b} coins for a trip. The bus costs {c} coins. "
            "How many coins are left after paying for the bus?",
            a * b - c,
        )
    c = rng.randint(5, a * b // 2)
    return (
        f"A printer prints {a} pages per minute. It runs for {b} minutes, a jam ruins {c} of those pages, "
        f"and then it runs for {d} more minutes. How many good pages are there?",
        a * b - c + a * d,
    )


def _mod(rng: random.Random) -> tuple[str, int]:
    a, b, c, m = rng.randint(113, 987), rng.randint(23, 97), rng.randint(100, 999), rng.randint(7, 19)
    return f"What is the remainder when {a} * {b} + {c} is divided by {m}?", (a * b + c) % m


def _digits(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(213, 987), rng.randint(113, 987)
    return f"What is the sum of the digits of {a} * {b}?", sum(int(ch) for ch in str(a * b))


def arithmetic(policy, n: int = 60, seed: int = 0, max_calls: int = 8) -> GeneratedBench:
    """Integer arithmetic: expressions, word problems, remainders, digit sums."""
    kinds = {"chain": _chain, "word": _word, "mod": _mod, "digits": _digits}
    return GeneratedBench(
        policy, "generated-arithmetic-v1", "arithmetic",
        "Every answer is an integer written with digits only: no words, units, commas or full stop.",
        _build(kinds, int(n), int(seed), "arithmetic"), int(max_calls),
    )


# -- strings ---------------------------------------------------------------------------------

def _letters(rng: random.Random, low: int, high: int, alphabet: str = string.ascii_lowercase) -> str:
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(low, high)))


def _reverse(rng: random.Random) -> tuple[str, str]:
    s = _letters(rng, 9, 14)
    return f"Reverse this string: {s}", s[::-1]


def _count(rng: random.Random) -> tuple[str, str]:
    s = _letters(rng, 26, 38, "abcde")
    letter = rng.choice("abcde")
    return f"How many times does the letter '{letter}' appear in this string: {s}", str(s.count(letter))


def _sort(rng: random.Random) -> tuple[str, str]:
    words = sorted({_letters(rng, 4, 7) for _ in range(rng.randint(6, 8))})
    shuffled = words[:]
    while shuffled == words:
        rng.shuffle(shuffled)
    return f"Sort these words alphabetically: {', '.join(shuffled)}", ", ".join(words)


def _caesar(rng: random.Random) -> tuple[str, str]:
    s, k = _letters(rng, 8, 12), rng.randint(2, 11)
    shifted = "".join(chr((ord(ch) - 97 + k) % 26 + 97) for ch in s)
    return (
        f"Shift every letter of this string forward by {k} positions in the alphabet, "
        f"wrapping from z to a: {s}",
        shifted,
    )


def _vowels(rng: random.Random) -> tuple[str, str]:
    s = _letters(rng, 12, 18, "aeiou" * 3 + "bcdfghklmnprst")
    while not s.strip("aeiou"):
        s = _letters(rng, 12, 18, "aeiou" * 3 + "bcdfghklmnprst")
    return f"Remove every vowel (a, e, i, o, u) from this string: {s}", "".join(ch for ch in s if ch not in "aeiou")


def strings(policy, n: int = 60, seed: int = 0, max_calls: int = 8) -> GeneratedBench:
    """Letter-level string manipulation: reverse, count, sort, shift, delete."""
    kinds = {"reverse": _reverse, "count": _count, "sort": _sort, "caesar": _caesar, "vowels": _vowels}
    return GeneratedBench(
        policy, "generated-strings-v1", "string-manipulation",
        "Strings are returned bare, without quotes; a count is digits only; a list of words is "
        "joined with a comma and a space.",
        _build(kinds, int(n), int(seed), "strings"), int(max_calls),
    )
