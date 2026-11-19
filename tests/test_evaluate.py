import time

import pytest

from crsi import LLMError, TaskResult, evaluate


class Bench:
    name = "b"

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def tasks(self, split):
        return ["t1", "t2"] if split == "evolve" else []

    def run(self, harness_dir, task_id, rollout):
        return self.behaviour(task_id, rollout)


def test_scores_are_means_over_rollouts_and_tasks(tmp_path):
    # t1 passes two rollouts of four, t2 passes one of four.
    def behaviour(task, rollout):
        ok = rollout < (2 if task == "t1" else 1)
        return TaskResult(task, rollout, float(ok), ok)

    report = evaluate(Bench(behaviour), tmp_path, "evolve", rollouts=4, workers=3, timeout=None)
    assert report.task_scores() == {"t1": 0.5, "t2": 0.25}
    assert report.score == pytest.approx(37.5) and (report.passed, report.total) == (3, 8)
    assert report.solved(0.5) == {"t1"} and report.solved(0.25) == {"t1", "t2"}
    report.save(tmp_path / "r.json")
    assert type(report).load(tmp_path / "r.json").to_dict() == report.to_dict()


def test_a_crashing_harness_is_a_failed_trial_with_its_traceback(tmp_path):
    def behaviour(task, rollout):
        if task == "t1":
            raise KeyError("answer")
        return "not a TaskResult"

    report = evaluate(Bench(behaviour), tmp_path, "evolve", rollouts=1, timeout=None)
    crashed, wrong_type = report.results
    assert not crashed.passed and "KeyError" in crashed.trace and crashed.error == "KeyError: 'answer'"
    assert not wrong_type.passed and "not a TaskResult" in wrong_type.error
    assert report.score == 0.0


def test_a_hung_task_times_out_and_fails(tmp_path):
    def behaviour(task, rollout):
        if task == "t1":
            time.sleep(5)
        return TaskResult(task, rollout, 1.0, True)

    started = time.monotonic()
    report = evaluate(Bench(behaviour), tmp_path, "evolve", rollouts=1, timeout=0.2)
    assert time.monotonic() - started < 3
    assert [r.passed for r in report.results] == [False, True] and "timed out" in report.results[0].error


def test_an_unreachable_model_stops_the_evaluation(tmp_path):
    def behaviour(task, rollout):
        raise LLMError("connection refused")

    with pytest.raises(LLMError):
        evaluate(Bench(behaviour), tmp_path, "evolve", rollouts=1, timeout=1.0)
    with pytest.raises(ValueError):
        evaluate(Bench(behaviour), tmp_path, "heldout")
