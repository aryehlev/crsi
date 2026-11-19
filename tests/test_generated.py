from pathlib import Path

import pytest

from crsi import LLMError, evaluate
from crsi.benchmarks import load_benchmark
from crsi.benchmarks.generated import arithmetic, strings

SEED_HARNESS = Path(__file__).parent.parent / "examples" / "seed_harness"

EXTRACTING = '''
def run(task, llm):
    reply = llm.complete("Answer with the bare answer only.", task["prompt"])
    return reply.split("The answer is ", 1)[-1].rstrip(".")
'''


class Oracle:
    """A policy that knows every answer but wraps it in a sentence."""

    name = "oracle"

    def __init__(self):
        self.answers = {}

    def learn(self, bench):
        self.answers.update({bench._prompt[t]: bench._answer[t] for t in bench._prompt})
        return bench

    def complete(self, system, user):
        return f"The answer is {self.answers[user]}."


@pytest.mark.parametrize("make, kinds", [(arithmetic, 4), (strings, 5)])
def test_generation_is_deterministic_and_split_in_half(make, kinds):
    a, b, other = make(Oracle()), make(Oracle()), make(Oracle(), seed=1)
    assert a._prompt == b._prompt and a._answer == b._answer and a._prompt != other._prompt
    evolve, heldout = a.tasks("evolve"), a.tasks("heldout")
    assert (len(evolve), len(heldout)) == (30, 30) and not set(evolve) & set(heldout)
    assert len({t.rsplit("-", 1)[0] for t in evolve}) == kinds == len({t.rsplit("-", 1)[0] for t in heldout})
    assert len(make(Oracle(), n=24).tasks("evolve")) == 12
    assert set(a.get(evolve[0])) == {"id", "prompt"}  # the harness never receives the answer
    assert a.policy == "oracle" and "run(task, llm) -> str" in a.interface


def test_answers_are_right():
    bench = arithmetic(Oracle(), n=8)
    prompt = bench._prompt["chain-0"]
    assert int(bench._answer["chain-0"]) == eval(prompt.split(": ", 1)[1])
    text = strings(Oracle(), n=10)
    assert text._answer["reverse-0"] == text._prompt["reverse-0"].split(": ", 1)[1][::-1]
    assert text._answer["sort-0"].split(", ") == sorted(text._prompt["sort-0"].split(": ", 1)[1].split(", "))
    assert text.grade("reverse-0", f"  {text._answer['reverse-0']}\n") and not text.grade("reverse-0", None)


@pytest.mark.parametrize("make", [arithmetic, strings])
def test_the_seed_harness_fails_on_format_and_a_better_harness_passes(make, tmp_path):
    oracle = Oracle()
    bench = oracle.learn(make(oracle, n=10))
    seed = evaluate(bench, SEED_HARNESS, "evolve", rollouts=1, timeout=None)
    assert seed.score == 0.0
    trace = seed.results[0].trace
    assert "--- model call 1 ---" in trace and "harness returned: 'The answer is " in trace and "result: fail" in trace
    (tmp_path / "harness.py").write_text(EXTRACTING)
    assert bench.validate(tmp_path) is None
    assert evaluate(bench, tmp_path, "heldout", rollouts=2, timeout=None).score == 100.0


def test_broken_harnesses_fail_tasks_and_are_caught_by_validation(tmp_path):
    oracle = Oracle()
    bench = oracle.learn(arithmetic(oracle, n=4, max_calls=2))
    (tmp_path / "harness.py").write_text("def run(task, llm):\n    return [llm.complete('s', task['prompt']) for _ in range(3)]\n")
    result = bench.run(tmp_path, "chain-0", 0)
    assert not result.passed and "call budget exceeded" in result.error and "harness raised" in result.trace
    no_run = tmp_path / "no_run"
    no_run.mkdir()
    (no_run / "harness.py").write_text("X = 1\n")
    assert bench.validate(no_run) == "harness.py does not define run(task, llm)"
    (no_run / "harness.py").write_text("import module_that_does_not_exist\n")
    assert bench.validate(no_run).startswith("harness.py does not import: ModuleNotFoundError")
    (no_run / "harness.py").write_text("def run(task, llm):\n    return 42\n")
    assert "harness returned int, not a string" in bench.run(no_run, "chain-0", 0).trace


def test_an_unreachable_policy_model_is_not_blamed_on_the_harness(tmp_path):
    class Down:
        name = "down"

        def complete(self, system, user):
            raise LLMError("connection refused")

    (tmp_path / "harness.py").write_text("def run(task, llm):\n    try:\n        return llm.complete('s', 'u')\n    except Exception:\n        return 'fallback'\n")
    with pytest.raises(LLMError):
        arithmetic(Down(), n=4).run(tmp_path, "chain-0", 0)


def test_load_benchmark_by_name_and_by_factory():
    assert load_benchmark("strings", Oracle(), n=6).name == "generated-strings-v1"
    toy = load_benchmark("toybench:factory", Oracle(), name="toy-Z", kinds="strip+lower")
    assert (toy.name, toy.kinds, toy.policy) == ("toy-Z", ("strip", "lower"), "oracle")
    with pytest.raises(ValueError, match="module:factory"):
        load_benchmark("nonsense", Oracle())
