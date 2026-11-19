import json
import re
import sys
from pathlib import Path

import pytest
from toybench import harness_with

from crsi import SIM
from crsi.benchmarks.generated import arithmetic, strings
from crsi.cli import main

ROOT = Path(__file__).parent.parent
SEED = str(ROOT / "examples" / "seed_harness")

EXTRACT = '''import re


def run(task, llm):
    reply = llm.complete("You are a helpful assistant.", task["prompt"])
    found = re.search(r"The answer is (.*)\\.$", reply.strip())
    return found.group(1)STRIP if found else reply
'''


def scripted_models(server) -> None:
    """One fake server plays all three roles: policy, proposer and selector."""
    answers = {}
    for bench in (arithmetic(None, n=8), strings(None, n=10)):
        answers.update({bench._prompt[t]: bench._answer[t] for t in bench._prompt})

    def reply(system: str, user: str) -> str:
        if system.startswith("You are a harness engineer"):
            cites = "## P1" in system
            failing = re.findall(r"^(\S+) \| 0\.00 \|", user, re.M)
            manifest = {
                "problem_id": "P1" if cites else None,
                "root_cause": "" if cites else "returns the model's whole sentence instead of the bare answer",
                "what": "extract the answer from the model's sentence",
                "why": "the grader compares the bare answer exactly",
                "predicted": failing,
            }
            harness = EXTRACT.replace("STRIP", ".strip()" if "re.search" in user else "")
            return f'<manifest>\n{json.dumps(manifest)}\n</manifest>\n<file path="harness.py">\n{harness}</file>'
        if system.startswith("You are the selector"):
            return '{"reject": []}' if "you screen" in system else '{"choice": 0, "reason": "fixes the format failures"}'
        prompt = next(p for p in answers if p in user)
        return f"The answer is {answers[prompt]}."

    server.reply = reply


def crsi(*args: str) -> int:
    return main([str(a) for a in args])


SMALL = ["--rounds", "1", "--candidates", "2", "--rollouts", "1", "--quiet",
         "--policy", "openai:local-policy", "--optimizer", "anthropic:claude-sonnet-5-5"]


def test_quickstart_end_to_end_against_a_fake_model_server(tmp_path, model_server, capsys):
    scripted_models(model_server)
    src = tmp_path / "src"
    assert crsi("run", "--bench", "arithmetic", "--bench-arg", "n=8", "--harness", SEED, "--out", src, *SMALL) == 0
    assert "src: held-out 0.00 -> 100.00 (+100.00); 1 of 1 rounds kept an edit" in capsys.readouterr().out
    record = json.loads((src / "rounds" / "r01" / "round.json").read_text())
    assert record["selected"] == 0 and record["selection_reason"] == "fixes the format failures"

    arms = ["cold_start", "harness_transfer", "problem_only", "full_sim"]
    for arm in arms:
        assert crsi("continue", "--bench", "strings", "--bench-arg", "n=10", "--from", src, "--arm", arm,
                    "--seed-harness", SEED, "--out", tmp_path / arm, *SMALL) == 0
    out = capsys.readouterr().out
    assert "cold_start: held-out 0.00 -> 100.00" in out and "full_sim: held-out 100.00 -> 100.00" in out

    assert crsi("report", src, *[tmp_path / arm for arm in arms]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].split() == ["run", "arm", "benchmark", "policy", "kept", "evolve0", "evolve", "heldout0", "heldout", "gain"]
    assert table[1].split() == ["src", "source", "generated-arithmetic-v1", "openai:local-policy", "1/1", "0.00", "100.00", "0.00", "100.00", "+100.00"]
    assert table[5].split()[:3] == ["full_sim", "full_sim", "generated-strings-v1"] and len(table) == 6

    assert crsi("show", tmp_path / "full_sim") == 0
    shown = capsys.readouterr().out
    assert shown.startswith("Carried from run 'src' (domain: arithmetic) into run 'full_sim'.")
    assert "attempt 1: round 1, sample 0 (carried from run 'src'); accepted; domain=arithmetic" in shown
    assert "attempt 2: round 1, sample 0; accepted; domain=string-manipulation, benchmark=generated-strings-v1" in shown
    assert crsi("show", tmp_path / "full_sim", "--round", "1", "--json") == 0
    assert json.loads(capsys.readouterr().out)["carried_from"]["arm"] == "src"
    assert SIM.load(tmp_path / "problem_only" / "sim.json").attempt_count() == 0
    assert crsi("show", tmp_path / "cold_start") == 0 and "has no memory" in capsys.readouterr().out

    # The policy only ever saw task prompts; the optimizer went to the other endpoint.
    paths = {r["path"] for r in model_server.requests}
    assert paths == {"/v1/chat/completions", "/v1/messages"}
    assert all(r["payload"]["model"] == "local-policy" for r in model_server.requests if r["path"] == "/v1/chat/completions")

    # Repeating a finished command is a no-op.
    before = len(model_server.requests)
    assert crsi("run", "--bench", "arithmetic", "--bench-arg", "n=8", "--harness", SEED, "--out", src, *SMALL) == 0
    assert len(model_server.requests) == before


def test_duplicate_proposals_are_screened_in_the_end_to_end_run(tmp_path, model_server):
    scripted_models(model_server)
    assert crsi("run", "--bench", "arithmetic", "--bench-arg", "n=8", "--harness", SEED, "--out", tmp_path / "src", *SMALL) == 0
    record = json.loads((tmp_path / "src" / "rounds" / "r01" / "round.json").read_text())
    assert [(c["status"], c["reason"]) for c in record["candidates"]] == [("evaluated", ""), ("screened_out", "duplicate of c0")]


STUB_AGENT = f'''
import json, pathlib
pathlib.Path("harness/harness.py").write_text({harness_with("strip")!r})
pathlib.Path("change_manifest.json").write_text(json.dumps(
    {{"problem_id": None, "root_cause": "no strip handler", "what": "add one", "why": "padding", "predicted": ["strip-0"]}}))
'''


def test_an_agent_command_with_greedy_selection_needs_no_optimizer_model(tmp_path, model_server, seed_harness, capsys):
    stub = tmp_path / "agent.py"
    stub.write_text(STUB_AGENT)
    code = crsi("run", "--bench", "toybench:factory", "--bench-arg", "kinds=strip+reverse", "--bench-arg", "per_kind=1",
                "--harness", seed_harness, "--out", tmp_path / "agent_run", "--name", "by-agent", "--policy", "openai:unused",
                "--proposer-cmd", f"{sys.executable} {stub}", "--greedy", "--rounds", "1", "--candidates", "1", "--rollouts", "1", "--quiet")
    assert code == 0 and "by-agent: held-out 0.00 -> 50.00" in capsys.readouterr().out
    assert model_server.requests == []
    assert (tmp_path / "agent_run" / "rounds" / "r01" / "c0" / "TASK.md").is_file()
    attempt = SIM.load(tmp_path / "agent_run" / "sim.json").problems[0].attempts[0]
    assert (attempt.run, attempt.fixed, attempt.passed, attempt.total) == ("by-agent", ["strip-0"], 1, 2)


def test_paired_takes_run_directories_or_plain_scores(tmp_path, source_run, capsys):
    assert crsi("paired", "--treatment", "21", "24", "22.5", "--control", "19", "20", "21") == 0
    out = capsys.readouterr().out
    assert "pairs: 3 (treatment better in 3)" in out and "mean difference: +2.50 points" in out and "paired t(2) = " in out
    assert crsi("paired", "--treatment", source_run, "50", "--control", "30", "40") == 0  # the run's held-out is 33.33
    assert "mean difference: +6.67 points" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="paired by position"):
        crsi("paired", "--treatment", "1", "2", "--control", "1")
    with pytest.raises(SystemExit, match="neither a number nor a finished run"):
        crsi("paired", "--treatment", tmp_path / "nowhere", "2", "--control", "1", "2")


def test_usage_errors_are_reported_without_a_traceback(tmp_path, model_server, seed_harness, capsys):
    base = ["--harness", seed_harness, "--out", tmp_path / "x", "--policy", "openai:p", "--quiet"]
    with pytest.raises(SystemExit, match="--optimizer"):
        crsi("run", "--bench", "arithmetic", *base)
    with pytest.raises(SystemExit, match="--greedy"):
        crsi("run", "--bench", "arithmetic", "--proposer-cmd", "true", *base)
    with pytest.raises(SystemExit, match="key=value"):
        crsi("run", "--bench", "arithmetic", "--bench-arg", "n", "--greedy", "--proposer-cmd", "true", *base)
    assert crsi("run", "--bench", "no-such-bench", "--greedy", "--proposer-cmd", "true", *base) == 1
    assert "unknown benchmark" in capsys.readouterr().err
    assert crsi("continue", "--bench", "arithmetic", "--arm", "full_sim", "--greedy", "--proposer-cmd", "true", *base[2:]) == 1
    assert "pass source_dir" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="not a crsi run directory"):
        crsi("report", tmp_path)
    with pytest.raises(SystemExit, match="no memory found"):
        crsi("show", tmp_path / "missing")
