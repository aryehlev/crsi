import json

import pytest
from helpers import ALL_A, ROOT_CAUSE, ScriptedLLM, ScriptedProposer, config, make_source, step
from toybench import SEED, FakePolicy, ToyBench, harness_with

from crsi import SIM, GreedySelector, LLMError, LLMProposer, RunError, run_source
from crsi.core import read_json
from crsi.harness import read_files


def run(tmp_path, seed, plan, selector=None, bench=None, **overrides):
    proposer = plan if hasattr(plan, "propose") else ScriptedProposer(plan)
    result = run_source(bench or ToyBench(FakePolicy()), seed, proposer, selector or GreedySelector(), config(tmp_path, "src", **overrides))
    return result, tmp_path / "src"


def test_an_accepted_edit_is_recorded_with_predicted_and_observed_outcomes(tmp_path, seed_harness):
    result = make_source(tmp_path, seed_harness)
    out = tmp_path / "src"
    assert (result["evolve_before"], round(result["evolve_after"], 2), result["accepted"]) == (0.0, 33.33, 1)
    assert (result["heldout_before"], round(result["heldout_after"], 2)) == (0.0, 33.33)
    assert read_files(out / "retained") == {"harness.py": harness_with("strip")}

    attempt = SIM.load(out / "sim.json").problems[0].attempts[0]
    assert (attempt.round, attempt.sample, attempt.accepted, attempt.backfilled) == (1, 1, True, False)
    assert attempt.predicted == ALL_A and attempt.fixed == ["strip-0", "strip-1"]
    assert attempt.missed == ["reverse-0", "reverse-1", "upper-0", "upper-1"]
    assert (attempt.passed, attempt.total, attempt.files) == (2, 6, ["harness.py"])
    assert (attempt.domain, attempt.benchmark, attempt.policy, attempt.run) == ("toy", "toy-A", "model-1", "src")

    record = read_json(out / "rounds" / "r01" / "round.json")
    assert record["selected"] == 1 and record["problem"] == "P1" and "+        return text.strip()" in record["selected_diff"]
    assert [c["status"] for c in record["candidates"]] == ["failed", "evaluated"]


def test_run_directory_layout(tmp_path, seed_harness):
    make_source(tmp_path, seed_harness)
    out = tmp_path / "src"
    for name in ("config.json", "state.json", "result.json", "sim.json", "retained/harness.py", "heldout_before.json",
                 "heldout_after.json", "start/harness/harness.py", "start/evolve.json", "start/sim.json",
                 "rounds/r01/round.json", "rounds/r01/sim.json", "rounds/r01/c1/manifest.json", "rounds/r01/c1/edit.diff",
                 "rounds/r01/c1/harness/harness.py", "rounds/r01/c1/eval.json"):
        assert (out / name).is_file(), name
    manifest = read_json(out / "rounds" / "r01" / "c1" / "manifest.json")
    assert manifest["root_cause"] == ROOT_CAUSE and manifest["files"] == ["harness.py"]  # written before evaluation
    assert SIM.load(out / "start" / "sim.json").problems == []


def test_when_nothing_is_accepted_harness_and_memory_stay_as_they_were(tmp_path, seed_harness):
    # upper is not a kind of this benchmark, so the edit changes no score and is not kept.
    plan = {(1, 0): step("handle upper", ("upper",), [], root_cause="no upper handler")}
    result, out = run(tmp_path, seed_harness, plan, bench=ToyBench(FakePolicy(), kinds=("strip", "reverse")))
    assert result["accepted"] == 0 and result["problems"] == 0
    assert read_files(out / "retained") == {"harness.py": SEED}
    assert read_json(out / "rounds" / "r01" / "round.json")["selected"] is None
    assert SIM.load(out / "sim.json").to_dict() == SIM().to_dict()


def test_static_screening_happens_before_any_evaluation(tmp_path, seed_harness):
    class Checked(ToyBench):
        def validate(self, harness_dir):
            return "uses eval" if "eval(" in (harness_dir / "harness.py").read_text() else None

    plan = {
        (1, 0): {"manifest": {"what": "nothing", "root_cause": "x"}},  # empty
        (1, 1): step("strip", ("strip",), [], root_cause="x"),
        (1, 2): step("strip again", ("strip",), [], root_cause="x"),  # duplicate of c1
        (1, 3): {"manifest": {"what": "broken", "root_cause": "x"}, "files": {"harness.py": "def run(:\n"}},
        (1, 4): step("cites nothing", ("reverse",), [], problem_id="P7"),
        (1, 5): {"manifest": {"what": "eval", "root_cause": "x"}, "files": {"harness.py": SEED + "# eval(x)\n"}},
        (1, 6): RuntimeError("boom"),
    }
    _, out = run(tmp_path, seed_harness, plan, bench=Checked(FakePolicy()), candidates=8)
    cands = read_json(out / "rounds" / "r01" / "round.json")["candidates"]
    reasons = {c["index"]: (c["status"], c["reason"]) for c in cands}
    assert reasons[0] == ("screened_out", "empty: the proposal does not change the harness")
    assert reasons[1][0] == "evaluated"
    assert reasons[2] == ("screened_out", "duplicate of c1")
    assert reasons[3][0] == "screened_out" and reasons[3][1].startswith("syntax error: harness.py:1")
    assert reasons[4] == ("screened_out", "the manifest cites 'P7', which is not in the memory, names no root cause")
    assert reasons[5] == ("screened_out", "benchmark check: uses eval")
    assert reasons[6] == ("failed", "proposer error: RuntimeError: boom")
    assert reasons[7] == ("failed", "unusable proposal: nothing to propose")
    assert [c["index"] for c in cands if c["score"] is not None] == [1]


def test_the_optimizer_never_sees_held_out_tasks(tmp_path, seed_harness):
    reply = '<manifest>{"problem_id": null, "root_cause": "no strip handler", "what": "strip", "why": "w", "predicted": ["strip-0", "strip-3"]}</manifest>\n<file path="harness.py">\n' + harness_with("strip") + "</file>"
    llm = ScriptedLLM(reply)
    bench = ToyBench(FakePolicy())
    _, out = run(tmp_path, seed_harness, LLMProposer(llm), bench=bench, rounds=2, candidates=1)
    prompts = "\n".join(system + user for system, user in llm.calls)
    assert len(llm.calls) == 2 and "strip-0" in prompts
    assert not [task for task in bench.tasks("heldout") if task in prompts]
    assert "Evolve score: 33.33" in llm.calls[1][1] and "heldout" not in prompts.lower()
    # A predicted id that is not an evolve task is dropped, not scored against held-out.
    manifest = read_json(out / "rounds" / "r01" / "c0" / "manifest.json")
    assert manifest["predicted"] == ["strip-0"] and manifest["ignored_predictions"] == ["strip-3"]


def test_history_of_earlier_rounds_reaches_the_proposer(tmp_path, seed_harness):
    plan = {
        (1, 0): step("add a strip handler", ("strip",), ["strip-0"], root_cause=ROOT_CAUSE),
        (2, 0): step("add a reverse handler", ("strip", "reverse"), ["reverse-0"], problem_id="P1"),
    }
    proposer = ScriptedProposer(plan)
    result, out = run(tmp_path, seed_harness, proposer, rounds=2, candidates=1)
    first, second = proposer.contexts
    assert first.history == [] and (first.round, second.round) == (1, 2)
    assert second.history[0]["selected"] == 0 and second.harness == {"harness.py": harness_with("strip")}
    assert second.evolve.solved() == {"strip-0", "strip-1"}
    assert [len(p.attempts) for p in SIM.load(out / "sim.json").problems] == [2] and result["accepted"] == 2
    assert len(SIM.load(out / "rounds" / "r01" / "sim.json").problems[0].attempts) == 1


def test_a_selected_edit_is_recorded_even_when_it_regresses(tmp_path, seed_harness):
    class TakeFirst(GreedySelector):
        def select(self, ctx, candidates):
            return candidates[0].index, "forced"

    plan = {
        (1, 0): step("strip", ("strip",), ["strip-0"], root_cause="no strip handler"),
        (2, 0): step("drop strip, add upper", ("upper",), ["upper-0", "strip-0"], root_cause="no upper handler"),
    }
    result, out = run(tmp_path, seed_harness, plan, selector=TakeFirst(), rounds=2, candidates=1)
    second = SIM.load(out / "sim.json").problems[1].attempts[0]
    assert second.accepted and second.fixed == ["upper-0"] and second.missed == ["strip-0"] and second.passed == 2
    assert read_json(out / "rounds" / "r02" / "round.json")["candidates"][0]["regressed_tasks"] == ["strip-0", "strip-1"]


def test_record_evaluated_also_logs_rejected_edits(tmp_path, seed_harness):
    plan = {
        (1, 0): step("add a strip handler", ("strip",), ["strip-0"], root_cause=ROOT_CAUSE),
        (1, 1): step("add strip and reverse handlers", ("strip", "reverse"), ["reverse-0"], root_cause=ROOT_CAUSE.upper()),
    }
    _, out = run(tmp_path, seed_harness, plan, record="evaluated")
    sim = SIM.load(out / "sim.json")
    assert len(sim) == 1 and sim.aliases == {"P2": "P1"}  # same text, merged
    assert [(a.sample, a.accepted) for a in sim.problems[0].attempts] == [(0, False), (1, True)]
    assert "evaluated, not accepted" in sim.render()
    _, default = run(tmp_path / "again", seed_harness, plan)
    assert SIM.load(default / "sim.json").attempt_count() == 1


def test_a_finished_run_is_not_repeated_and_can_be_extended(tmp_path, seed_harness):
    plan = {
        (1, 0): step("add a strip handler", ("strip",), ["strip-0"], root_cause=ROOT_CAUSE),
        (2, 0): step("add a reverse handler", ("strip", "reverse"), ["reverse-0"], problem_id="P1"),
    }
    first = ScriptedProposer(plan)
    result, out = run(tmp_path, seed_harness, first, candidates=1)
    again = ScriptedProposer(plan)
    assert run(tmp_path, seed_harness, again, candidates=1)[0] == result and again.contexts == []
    extended, _ = run(tmp_path, seed_harness, again, candidates=1, rounds=2)
    assert [c.round for c in again.contexts] == [2] and extended["rounds"] == 2
    assert round(extended["heldout_after"], 2) == 66.67 and extended["heldout_before"] == 0.0


def test_an_interrupted_run_resumes_after_the_last_completed_round(tmp_path, seed_harness):
    plan = {
        (1, 0): step("add a strip handler", ("strip",), ["strip-0"], root_cause=ROOT_CAUSE),
        (2, 0): LLMError("connection lost"),
    }
    with pytest.raises(LLMError):
        run(tmp_path, seed_harness, plan, rounds=3, candidates=1)
    out = tmp_path / "src"
    state = read_json(out / "state.json")
    assert (state["completed"], state["status"]) == (1, "running") and not (out / "result.json").exists()
    (out / "retained" / "harness.py").write_text("corrupted by the crash")

    plan[(2, 0)] = step("add a reverse handler", ("strip", "reverse"), ["reverse-0"], problem_id="P1")
    resumed = ScriptedProposer(plan)
    result, _ = run(tmp_path, seed_harness, resumed, rounds=3, candidates=1)
    assert [c.round for c in resumed.contexts] == [2, 3]  # round 1 is not redone
    assert result["accepted"] == 2 and read_files(out / "retained") == {"harness.py": harness_with("strip", "reverse")}
    assert len(SIM.load(out / "sim.json").problems[0].attempts) == 2


def test_a_run_directory_cannot_be_reused_with_other_settings(tmp_path, seed_harness):
    make_source(tmp_path, seed_harness)
    with pytest.raises(RunError, match="rollouts: 1 -> 3"):
        make_source(tmp_path, seed_harness, rollouts=3)
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "notes.txt").write_text("mine")
    with pytest.raises(RunError, match="not a crsi run directory"):
        run_source(ToyBench(FakePolicy()), seed_harness, ScriptedProposer({}), GreedySelector(), config(tmp_path, "stray"))
    assert json.loads((tmp_path / "src" / "config.json").read_text())["rollouts"] == 1
