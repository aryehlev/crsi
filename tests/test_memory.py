import json

import pytest

from crsi import SIM, Attempt


def attempt(round=1, sample=0, run="src", **kw):
    base = dict(round=round, sample=sample, accepted=True, what="edit", why="because", files=["harness.py"],
                predicted=["a", "b"], fixed=["a"], missed=["b"], passed=3, total=6,
                domain="toy", benchmark="toy-A", policy="model-1", run=run)
    base.update(kw)
    return Attempt(**base)


def test_record_opens_a_problem_per_new_root_cause():
    sim = SIM()
    first = sim.record(attempt(round=2), root_cause="premature completion")
    second = sim.record(attempt(round=3), root_cause="answers in the wrong format")
    assert (first.id, first.opened_round, second.id) == ("P1", 2, "P2")
    assert sim.next_id == 3 and len(sim) == 2 and sim.attempt_count() == 2


def test_record_appends_to_the_cited_problem_and_keeps_earlier_attempts():
    sim = SIM()
    sim.record(attempt(what="first"), root_cause="premature completion")
    problem = sim.record(attempt(round=4, what="second"), problem_id="P1", root_cause="ignored when an id is cited")
    assert [a.what for a in problem.attempts] == ["first", "second"]
    assert len(sim) == 1 and problem.opened_round == 1


def test_identical_root_cause_text_is_merged_with_an_alias():
    sim = SIM()
    sim.record(attempt(what="first"), root_cause="Premature completion")
    sim.record(attempt(what="other"), root_cause="format")
    merged = sim.record(attempt(what="again"), root_cause="  premature   COMPLETION ")
    assert merged.id == "P1" and [a.what for a in merged.attempts] == ["first", "again"]
    assert sim.aliases == {"P3": "P1"} and sim.next_id == 4
    assert [p.id for p in sim.problems] == ["P1", "P2"]
    assert sim.get("P3") is merged  # a citation of the merged id still resolves
    assert sim.record(attempt(what="via alias"), problem_id="P3").id == "P1"


def test_unknown_citation_without_a_root_cause_is_an_error():
    with pytest.raises(ValueError):
        SIM().record(attempt(), problem_id="P9")


def test_stored_json_uses_the_papers_field_names(tmp_path):
    sim = SIM()
    sim.record(attempt(), root_cause="premature completion")
    sim.save(tmp_path / "sim.json")
    data = json.loads((tmp_path / "sim.json").read_text())
    assert list(data) == ["next_id", "carried_from", "aliases", "problems"]
    assert list(data["problems"][0]) == ["id", "name", "opened_round", "attempts"]
    paper_fields = ["round", "sample", "accepted", "backfilled", "what", "files", "predicted", "fixed", "missed", "passed", "total", "domain"]
    assert list(data["problems"][0]["attempts"][0])[: len(paper_fields)] == paper_fields
    assert data["problems"][0]["attempts"][0]["backfilled"] is False
    assert SIM.load(tmp_path / "sim.json").to_dict() == sim.to_dict()


def test_carry_keeps_the_full_record_and_marks_where_attempts_came_from():
    sim = SIM()
    sim.record(attempt(), root_cause="premature completion")
    carried = sim.carry("src", "toy", into="tgt")
    carried.record(attempt(run="tgt", benchmark="toy-B"), problem_id="P1")
    assert carried.carried_from == {"arm": "src", "domain": "toy", "into": "tgt"}
    assert sim.attempt_count() == 1  # the source memory is not touched
    text = carried.render()
    assert text.startswith("Carried from run 'src' (domain: toy) into run 'tgt'.")
    assert "attempt 1: round 1, sample 0 (carried from run 'src'); accepted; domain=toy, benchmark=toy-A, policy=model-1" in text
    assert "attempt 2: round 1, sample 0; accepted; domain=toy, benchmark=toy-B, policy=model-1" in text
    assert "predicted to fix (2): a, b" in text and "fixed (1): a; missed (1): b; evolve trials passed 3/6" in text


def test_problem_only_memory_carries_root_causes_and_never_stores_attempts():
    sim = SIM()
    sim.record(attempt(), root_cause="premature completion")
    bare = sim.carry("src", "toy", into="tgt", problems_only=True)
    assert bare.attempt_count() == 0 and [p.name for p in bare.problems] == ["premature completion"]
    bare.record(attempt(run="tgt"), problem_id="P1")
    bare.record(attempt(run="tgt", round=2), root_cause="a new diagnosis")
    assert bare.attempt_count() == 0 and [p.id for p in bare.problems] == ["P1", "P2"]
    text = bare.render()
    assert "Root causes only" in text and "## P2: a new diagnosis" in text and "what:" not in text
    assert SIM.from_dict(bare.to_dict()).problem_only is True


def test_empty_memory_renders_a_plain_statement():
    assert SIM().render() == "The memory is empty: no root cause has been recorded yet."
