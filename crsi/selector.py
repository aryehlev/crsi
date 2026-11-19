"""Selectors: S(H_t, F_t, C_t, F_t^C; M_t).

A selector is any object with

    screen(ctx, candidates) -> {candidate index: reason}      proposals to drop before evaluation
    select(ctx, candidates) -> (candidate index or None, reason)   at most one edit to keep

The paper gives no scoring rule. Here both steps are one model call each, reading the same
memory as the proposer, with a score-based fallback when the reply cannot be read.
"""

from __future__ import annotations

from . import prompts
from .core import Candidate, Context
from .llm import LLMRefusal
from .proposer import extract_json


def best_by_score(ctx: Context, candidates: list[Candidate]) -> tuple[int | None, str]:
    """The best evolve score, if it beats the current harness. Ties go to the smaller diff."""
    scored = [c for c in candidates if c.report is not None]
    if not scored:
        return None, "no candidate was evaluated"
    best = max(scored, key=lambda c: (c.report.score, -sum(c.diff_stat()), -c.index))
    if best.report.score > ctx.evolve.score:
        return best.index, f"best evolve score: {best.report.score:.2f} against {ctx.evolve.score:.2f}"
    return None, f"no candidate beat the current evolve score of {ctx.evolve.score:.2f}"


class GreedySelector:
    """No model: nothing is screened out, and the best evolve score wins if it is an improvement."""

    def screen(self, ctx: Context, candidates: list[Candidate]) -> dict[int, str]:
        return {}

    def select(self, ctx: Context, candidates: list[Candidate]) -> tuple[int | None, str]:
        return best_by_score(ctx, candidates)


class LLMSelector:
    def __init__(self, llm, diff_chars: int = 6000) -> None:
        self.llm = llm
        self.diff_chars = diff_chars

    def _ask(self, system: str, user: str) -> dict | None:
        try:
            return extract_json(self.llm.complete(system, user))
        except LLMRefusal:
            return None

    def screen(self, ctx: Context, candidates: list[Candidate]) -> dict[int, str]:
        if not candidates:
            return {}
        reply = self._ask(prompts.screen_system(ctx), prompts.selector_user(ctx, candidates, self.diff_chars))
        rejected = reply.get("reject") if reply else None
        if not isinstance(rejected, list):
            return {}  # unreadable: keep everything, evaluation will settle it
        valid = {c.index for c in candidates}
        out: dict[int, str] = {}
        for item in rejected:
            if not isinstance(item, dict):
                continue
            index = _index(item.get("candidate"))
            if index in valid:
                out[index] = " ".join(str(item.get("reason") or "rejected by the selector").split())
        return out

    def select(self, ctx: Context, candidates: list[Candidate]) -> tuple[int | None, str]:
        if not candidates:
            return None, "no candidate was evaluated"
        reply = self._ask(prompts.select_system(ctx), prompts.selector_user(ctx, candidates, self.diff_chars))
        if reply is not None and "choice" in reply:
            reason = " ".join(str(reply.get("reason") or "").split())
            if reply["choice"] is None:
                return None, reason or "the selector kept the current harness"
            index = _index(reply["choice"])
            if index in {c.index for c in candidates}:
                return index, reason or "chosen by the selector"
        choice, reason = best_by_score(ctx, candidates)
        return choice, f"selector reply unreadable, fell back to score: {reason}"


def _index(value) -> int | None:
    """Accept 2, "2" and "c2"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = value.strip().lower().removeprefix("candidate").strip().lstrip("c")
        if digits.isdigit():
            return int(digits)
    return None
