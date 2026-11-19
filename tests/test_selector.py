from pathlib import Path

from helpers import ScriptedLLM

from crsi import Candidate, Context, EvalReport, GreedySelector, LLMRefusal, LLMSelector, Manifest, TaskResult


def report(*passed: bool) -> EvalReport:
    return EvalReport("evolve", [TaskResult(f"t{i}", 0, float(ok), ok) for i, ok in enumerate(passed)])


def ctx(current: EvalReport) -> Context:
    return Context(run="r", round=1, rounds=1, candidates=3, benchmark="b", domain="d", policy="p", interface="i",
                   harness_dir=Path("."), harness={}, evolve=current, history=[], memory=None, memory_mode="none")


def cand(index: int, rep: EvalReport | None, diff_lines: int = 1) -> Candidate:
    c = Candidate(index, Path(f"c{index}"), manifest=Manifest(what=f"edit {index}", root_cause="cause"))
    c.report, c.diff = rep, "\n".join(["+x"] * diff_lines)
    c.status = "evaluated" if rep else "proposed"
    return c


def test_greedy_keeps_the_best_score_only_if_it_improves():
    current = ctx(report(True, False, False))
    choice, reason = GreedySelector().select(current, [cand(0, report(True, True, False)), cand(1, report(True, True, True))])
    assert choice == 1 and "100.00 against 33.33" in reason
    assert GreedySelector().select(current, [cand(0, report(True, False, False))])[0] is None  # a tie is not an improvement
    assert GreedySelector().select(current, [])[0] is None
    assert GreedySelector().screen(current, [cand(0, None)]) == {}


def test_greedy_breaks_score_ties_towards_the_smaller_edit():
    current = ctx(report(False, False))
    big, small = cand(0, report(True, False), diff_lines=40), cand(1, report(True, False), diff_lines=3)
    assert GreedySelector().select(current, [big, small])[0] == 1


def test_llm_screen_drops_named_candidates_and_keeps_all_when_unreadable():
    current = ctx(report(False))
    pool = [cand(0, None), cand(1, None), cand(2, None)]
    llm = ScriptedLLM('{"reject": [{"candidate": "c1", "reason": "hard-codes\\n answers"}, {"candidate": 7, "reason": "no such"}]}')
    assert LLMSelector(llm).screen(current, pool) == {1: "hard-codes answers"}
    system, user = llm.calls[0]
    assert "screen" in system and "## Candidate 2" in user and "what: edit 1" in user
    assert LLMSelector(ScriptedLLM("I cannot decide")).screen(current, pool) == {}
    assert LLMSelector(ScriptedLLM(LLMRefusal("declined"))).screen(current, pool) == {}


def test_llm_select_follows_the_model_including_a_choice_to_keep_the_current_harness():
    current = ctx(report(True, False))
    pool = [cand(0, report(True, True)), cand(1, report(False, False))]
    llm = ScriptedLLM('{"choice": 1, "reason": "simpler and more general"}')
    assert LLMSelector(llm).select(current, pool) == (1, "simpler and more general")  # may pick a regression
    assert "evolve score: 0.00 (current harness: 50.00)" in llm.calls[0][1]
    assert LLMSelector(ScriptedLLM('{"choice": null, "reason": "noise"}')).select(current, pool) == (None, "noise")


def test_llm_select_falls_back_to_score_when_the_reply_is_unusable():
    current = ctx(report(True, False))
    pool = [cand(0, report(True, True)), cand(1, report(False, False))]
    for reply in ("the first one looks good", '{"choice": 9}', '{"reason": "no choice"}'):
        choice, reason = LLMSelector(ScriptedLLM(reply)).select(current, pool)
        assert choice == 0 and reason.startswith("selector reply unreadable")
