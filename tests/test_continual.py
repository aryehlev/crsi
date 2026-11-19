import pytest
from helpers import ScriptedProposer, config, step
from toybench import SEED, FakePolicy, ToyBench, harness_with

from crsi import ARMS, SIM, GreedySelector, RunError, continue_rsi
from crsi.core import read_json
from crsi.harness import read_files

REVERSE = step("add a reverse handler", ("strip", "reverse"), ["reverse-0", "reverse-1"], problem_id="P1",
               root_cause="no reverse handler", why="reverse tasks come back unchanged")


def target(policy="model-1"):
    return ToyBench(FakePolicy(policy), name="toy-B", kinds=("strip", "reverse", "lower"))


def continue_arm(tmp_path, arm, source_run, seed_harness, plan=None, **overrides):
    proposer = ScriptedProposer(plan if plan is not None else {(1, 0): REVERSE})
    result = continue_rsi(target(), arm, proposer, GreedySelector(), config(tmp_path, arm, candidates=1, **overrides),
                          source_dir=source_run, seed_harness=seed_harness)
    return result, tmp_path / arm, proposer


def test_each_arm_starts_from_the_harness_and_memory_the_paper_defines(tmp_path, source_run, seed_harness):
    assert ARMS == ("cold_start", "harness_transfer", "problem_only", "full_sim")
    evolved = {"harness.py": harness_with("strip")}
    starts = {}
    for arm in ARMS:
        result, out, proposer = continue_arm(tmp_path, arm, source_run, seed_harness, rounds=0)
        starts[arm] = (read_files(out / "start" / "harness"), result["heldout_before"], (out / "start" / "sim.json").exists())
        assert result["arm"] == arm and result["source"] == "src" and result["heldout_after"] == result["heldout_before"]
    assert starts["cold_start"] == ({"harness.py": SEED}, 0.0, False)  # (H_0, none)
    assert starts["harness_transfer"] == (evolved, pytest.approx(33.33, abs=0.01), False)  # (H_S, none)
    assert starts["problem_only"][0] == evolved and starts["problem_only"][2]  # (H_S, M_S^c)
    assert starts["full_sim"][0] == evolved and starts["full_sim"][2]  # (H_S, M_S)
    bare, full = SIM.load(tmp_path / "problem_only" / "start" / "sim.json"), SIM.load(tmp_path / "full_sim" / "start" / "sim.json")
    assert [p.name for p in bare.problems] == [p.name for p in full.problems] and bare.attempt_count() == 0
    assert full.attempt_count() == 1 and full.carried_from == {"arm": "src", "domain": "toy", "into": "full_sim"}


def test_arms_without_memory_give_the_optimizer_none_and_record_nothing(tmp_path, source_run, seed_harness):
    for arm in ("cold_start", "harness_transfer"):
        result, out, proposer = continue_arm(tmp_path, arm, source_run, seed_harness)
        ctx = proposer.contexts[0]
        assert ctx.memory is None and ctx.memory_mode == "none"
        assert result["accepted"] == 1 and result["problems"] == 0 and not (out / "sim.json").exists()
        assert read_json(out / "rounds" / "r01" / "round.json")["problem"] is None


def test_problem_only_memory_gains_root_causes_but_never_attempts(tmp_path, source_run, seed_harness):
    plan = {
        (1, 0): REVERSE,
        (2, 0): step("add a lower handler", ("strip", "reverse", "lower"), ["lower-0"], root_cause="no lower handler"),
    }
    result, out, proposer = continue_arm(tmp_path, "problem_only", source_run, seed_harness, plan, rounds=2)
    first = proposer.contexts[0]
    assert first.memory_mode == "problems" and "add a strip handler" not in first.memory.render()
    sim = SIM.load(out / "sim.json")
    assert result["accepted"] == 2 and [p.id for p in sim.problems] == ["P1", "P2"]
    assert sim.attempt_count() == 0 and sim.problem_only and result["attempts"] == 0


def test_full_sim_extends_the_carried_record(tmp_path, continued_run):
    sim = SIM.load(continued_run / "sim.json")
    carried, new = sim.problems[0].attempts
    assert (carried.run, carried.benchmark, carried.fixed) == ("src", "toy-A", ["strip-0", "strip-1"])
    assert (new.run, new.benchmark, new.round, new.fixed, new.missed) == ("tgt", "toy-B", 1, ["reverse-0", "reverse-1"], [])
    assert (new.passed, new.total) == (4, 6) and sim.carried_from["into"] == "tgt"
    assert SIM.load(tmp_path / "src" / "sim.json").attempt_count() == 1  # the source run is left alone


def test_a_continued_run_can_be_continued_again_on_another_model(tmp_path, continued_run):
    plan = {(1, 0): step("add a lower handler", ("strip", "reverse", "lower"), ["lower-0", "lower-1"], problem_id="P1")}
    bench = ToyBench(FakePolicy("model-2"), name="toy-B", kinds=("strip", "reverse", "lower"))
    result = continue_rsi(bench, "full_sim", ScriptedProposer(plan), GreedySelector(), config(tmp_path, "gen3", candidates=1), source_dir=continued_run)
    sim = SIM.load(tmp_path / "gen3" / "sim.json")
    assert [(a.run, a.policy) for a in sim.problems[0].attempts] == [("src", "model-1"), ("tgt", "model-1"), ("gen3", "model-2")]
    assert sim.carried_from == {"arm": "tgt", "domain": "toy", "into": "gen3"} and result["heldout_after"] == 100.0
    text = sim.render()
    assert "(carried from run 'src')" in text and "(carried from run 'tgt')" in text and "sample 0; accepted; domain=toy, benchmark=toy-B, policy=model-2" in text


def test_continuation_rejects_inputs_it_cannot_use(tmp_path, source_run, seed_harness):
    args = (ScriptedProposer({}), GreedySelector())
    with pytest.raises(ValueError):
        continue_rsi(target(), "warm_start", *args, config(tmp_path, "x"), source_dir=source_run)
    with pytest.raises(RunError, match="seed_harness"):
        continue_rsi(target(), "cold_start", *args, config(tmp_path, "x"), source_dir=source_run)
    with pytest.raises(RunError, match="source_dir"):
        continue_rsi(target(), "full_sim", *args, config(tmp_path, "x"))
    with pytest.raises(RunError, match="not a finished run"):
        continue_rsi(target(), "full_sim", *args, config(tmp_path, "x"), source_dir=tmp_path / "seed")
    continue_arm(tmp_path, "harness_transfer", source_run, seed_harness, rounds=0)
    with pytest.raises(RunError, match="no memory to carry"):
        continue_rsi(target(), "full_sim", *args, config(tmp_path, "y"), source_dir=tmp_path / "harness_transfer")
