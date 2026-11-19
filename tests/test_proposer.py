import sys
from pathlib import Path

import pytest
from helpers import ScriptedLLM
from toybench import SEED

from crsi import SIM, Attempt, CommandProposer, Context, EvalReport, LLMProposer, LLMRefusal, ProposalError, TaskResult
from crsi.harness import copy_harness, read_files
from crsi.prompts import MEMORY_RULES
from crsi.proposer import extract_json, parse_proposal

REPLY = """I looked at the traces.

<manifest>
```json
{"problem_id": "P1", "root_cause": "", "what": "strip the answer", "why": "padding", "predicted": ["strip-0", "ghost-9"]}
```
</manifest>

<file path="harness.py">
```python
def run(task, llm):
    return task["text"].strip()
```
</file>
<file path='notes/readme.txt'>
two
lines
</file>
<delete path="old.py"/>
"""


def context(tmp_path: Path, memory=None, mode="full") -> Context:
    harness = tmp_path / "current"
    harness.mkdir(exist_ok=True)
    (harness / "harness.py").write_text(SEED)
    (harness / "old.py").write_text("X = 1\n")
    report = EvalReport("evolve", [
        TaskResult("strip-0", 0, 0.0, False, trace="input '  Ab0  ' returned '  Ab0  ' expected 'Ab0'"),
        TaskResult("upper-0", 0, 1.0, True, trace="fine"),
    ])
    return Context(run="tgt", round=2, rounds=3, candidates=2, benchmark="toy-B", domain="toy", policy="model-1",
                   interface="run(task, llm) -> str", harness_dir=harness, harness=read_files(harness), evolve=report,
                   history=[], memory=memory, memory_mode=mode)


def workdir(tmp_path: Path, ctx: Context) -> Path:
    work = tmp_path / "c0"
    work.mkdir()
    copy_harness(ctx.harness_dir, work / "harness")
    return work


def memory_with_one_attempt() -> SIM:
    sim = SIM()
    sim.record(Attempt(round=1, sample=0, accepted=True, what="add a strip handler", why="padding", domain="toy",
                       benchmark="toy-A", policy="model-1", run="src"), root_cause="no strip handler")
    return sim.carry("src", "toy", into="tgt")


def test_parse_proposal_reads_manifest_files_and_deletions():
    manifest, files, deletes = parse_proposal(REPLY)
    assert (manifest.problem_id, manifest.what, manifest.why) == ("P1", "strip the answer", "padding")
    assert manifest.predicted == ["strip-0", "ghost-9"]
    assert files == {"harness.py": 'def run(task, llm):\n    return task["text"].strip()\n', "notes/readme.txt": "two\nlines\n"}
    assert deletes == ["old.py"]


@pytest.mark.parametrize("reply", ["no tags at all", "<manifest>not json</manifest>", '<manifest>{"why": "no what"}</manifest>'])
def test_a_reply_without_a_usable_manifest_is_rejected(reply):
    with pytest.raises(ProposalError):
        parse_proposal(reply)


def test_extract_json_tolerates_prose_and_fences():
    assert extract_json('Sure: {"choice": 2, "reason": "x {y}"} done') == {"choice": 2, "reason": "x {y}"}
    assert extract_json("```json\n{\"reject\": []}\n```") == {"reject": []}
    assert extract_json("{broken} then {\"ok\": true}") == {"ok": True}
    assert extract_json("nothing here") is None


def test_llm_proposer_applies_the_edit_and_shows_the_whole_memory(tmp_path):
    ctx = context(tmp_path, memory_with_one_attempt())
    llm = ScriptedLLM(REPLY)
    work = workdir(tmp_path, ctx)
    manifest = LLMProposer(llm).propose(ctx, 0, work)
    assert manifest.problem_id == "P1"
    assert read_files(work / "harness") == {"harness.py": 'def run(task, llm):\n    return task["text"].strip()\n', "notes/readme.txt": "two\nlines\n"}
    assert (work / "proposal.txt").read_text() == REPLY
    system, user = llm.calls[0]
    assert ctx.memory.render() in system and all(rule in system for rule in MEMORY_RULES)
    assert "benchmark: toy-B" in system and "policy model (fixed): model-1" in system
    assert SEED.strip() in user and "strip-0 | 0.00 | 0/1" in user
    assert "expected 'Ab0'" in user and 'trace task="upper-0"' not in user  # traces of failures only


def test_without_memory_the_prompt_has_no_memory_section(tmp_path):
    ctx = context(tmp_path, memory=None, mode="none")
    llm = ScriptedLLM(REPLY)
    LLMProposer(llm).propose(ctx, 0, workdir(tmp_path, ctx))
    system = llm.calls[0][0]
    assert "Self-Improvement Memory" not in system and "problem_id" not in system and "root_cause" in system


def test_problem_only_prompt_lists_root_causes_without_outcomes(tmp_path):
    bare = memory_with_one_attempt().carry("src", "toy", into="tgt", problems_only=True)
    ctx = context(tmp_path, bare, mode="problems")
    llm = ScriptedLLM(REPLY)
    LLMProposer(llm).propose(ctx, 0, workdir(tmp_path, ctx))
    system = llm.calls[0][0]
    assert "## P1: no strip handler" in system and "add a strip handler" not in system and MEMORY_RULES[3] not in system


def test_llm_proposer_refuses_unsafe_paths_and_refusals(tmp_path):
    ctx = context(tmp_path)
    escape = REPLY.replace('path="harness.py"', 'path="../../outside.py"')
    with pytest.raises(ProposalError, match="unsafe path"):
        LLMProposer(ScriptedLLM(escape)).propose(ctx, 0, workdir(tmp_path, ctx))
    assert not (tmp_path / "outside.py").exists() and not (tmp_path.parent / "outside.py").exists()
    with pytest.raises(ProposalError, match="declined"):
        LLMProposer(ScriptedLLM(LLMRefusal("model declined the request"))).propose(ctx, 0, tmp_path / "c0")


STUB_AGENT = '''
import json, pathlib, sys
assert "TASK.md" in sys.argv[-1]
task = pathlib.Path("TASK.md").read_text()
assert "change_manifest.json" in task and "## P1: no strip handler" in task
assert pathlib.Path("feedback/summary.md").exists() and "expected 'Ab0'" in pathlib.Path("feedback/000_strip-0.txt").read_text()
if "--fail" in sys.argv:
    sys.exit("agent crashed")
pathlib.Path("harness/harness.py").write_text('def run(task, llm):\\n    return task["text"].strip()\\n')
pathlib.Path("change_manifest.json").write_text(json.dumps({"problem_id": "P1", "what": "strip", "why": "padding", "predicted": ["strip-0"]}))
print("edited harness.py")
'''


def test_an_agent_cli_can_be_the_proposer(tmp_path):
    stub = tmp_path / "agent.py"
    stub.write_text(STUB_AGENT)
    ctx = context(tmp_path, memory_with_one_attempt())
    work = workdir(tmp_path, ctx)
    manifest = CommandProposer(f"{sys.executable} {stub}").propose(ctx, 0, work)
    assert (manifest.problem_id, manifest.what, manifest.predicted) == ("P1", "strip", ["strip-0"])
    assert ".strip()" in (work / "harness" / "harness.py").read_text()
    assert "edited harness.py" in (work / "proposal.txt").read_text()


def test_an_agent_that_writes_no_manifest_is_an_unusable_proposal(tmp_path):
    stub = tmp_path / "agent.py"
    stub.write_text(STUB_AGENT)
    ctx = context(tmp_path, memory_with_one_attempt())
    with pytest.raises(ProposalError, match="did not write change_manifest.json.*agent crashed"):
        CommandProposer(f"{sys.executable} {stub} --fail {{prompt}}").propose(ctx, 0, workdir(tmp_path, ctx))
    with pytest.raises(ProposalError, match="could not start"):
        CommandProposer("no-such-agent-binary-xyz").propose(ctx, 0, tmp_path / "c0")
