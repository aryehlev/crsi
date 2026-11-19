"""Proposers: P(H_t, F_t, D_t; M_t).

A proposer is any object with

    propose(ctx: Context, sample: int, workdir: Path) -> Manifest

When it is called, `workdir/harness/` already holds a copy of the current harness. The proposer
edits that copy and returns the change manifest. It raises ProposalError for a proposal that
cannot be used.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

from . import prompts
from .core import Context, Manifest, ProposalError
from .harness import apply_edits
from .llm import LLMRefusal

_MANIFEST = re.compile(r"<manifest>(.*?)</manifest>", re.S)
_FILE = re.compile(r"<file\s+path=[\"']([^\"']+)[\"']\s*>\n?(.*?)\n?</file>", re.S)
_DELETE = re.compile(r"<delete\s+path=[\"']([^\"']+)[\"']\s*/?>")
_FENCE = re.compile(r"\A\s*```[\w+-]*\n(.*?)\n?```\s*\Z", re.S)


def extract_json(text: str):
    """The first JSON object in a model reply, or None. Tolerates prose and code fences around it."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def parse_proposal(reply: str) -> tuple[Manifest, dict[str, str], list[str]]:
    """Split a reply into (manifest, {path: content}, [deleted paths])."""
    match = _MANIFEST.search(reply)
    data = extract_json(match.group(1)) if match else None
    if data is None:
        raise ProposalError("the reply has no readable <manifest> JSON")
    manifest = Manifest.from_dict(data)
    files: dict[str, str] = {}
    for path, body in _FILE.findall(reply):
        fenced = _FENCE.match(body)
        body = fenced.group(1) if fenced else body
        files[path.strip()] = body if body.endswith("\n") or not body else body + "\n"
    deletes = [path.strip() for path in _DELETE.findall(reply)]
    return manifest, files, deletes


class LLMProposer:
    """One model call per candidate, with the whole harness and the whole memory in the prompt."""

    def __init__(self, llm, trace_chars: int = 1500, max_traces: int = 20, history_diff_chars: int = 4000) -> None:
        self.llm = llm
        self.trace_chars = trace_chars
        self.max_traces = max_traces
        self.history_diff_chars = history_diff_chars

    def propose(self, ctx: Context, sample: int, workdir: Path) -> Manifest:
        system = prompts.proposer_system(ctx)
        user = prompts.proposer_user(ctx, self.trace_chars, self.max_traces, self.history_diff_chars)
        try:
            reply = self.llm.complete(system, user)
        except LLMRefusal as exc:
            raise ProposalError(str(exc)) from None
        (workdir / "proposal.txt").write_text(reply, encoding="utf-8")
        manifest, files, deletes = parse_proposal(reply)
        try:
            apply_edits(workdir / "harness", files, deletes)
        except (ValueError, OSError) as exc:
            raise ProposalError(str(exc)) from None
        return manifest


AGENT_PROMPT = "Read TASK.md in the current directory and carry out the task it describes."


class CommandProposer:
    """Hand the proposal to an agent CLI (Claude Code, Codex, Gemini CLI, OpenCode, ...).

    The command runs in a scratch directory holding TASK.md, `harness/` to edit in place and
    `feedback/` with full traces. It must write `change_manifest.json` there. The instruction is
    appended as the last argument, or substituted for `{prompt}` if the command contains it.
    """

    def __init__(self, command: str, timeout: float = 1800.0, prompt: str = AGENT_PROMPT) -> None:
        self.command = command
        self.timeout = timeout
        self.prompt = prompt

    def propose(self, ctx: Context, sample: int, workdir: Path) -> Manifest:
        (workdir / "TASK.md").write_text(prompts.agent_task(ctx), encoding="utf-8")
        self._write_feedback(ctx, workdir / "feedback")
        argv = shlex.split(self.command)
        if any("{prompt}" in arg for arg in argv):
            argv = [arg.replace("{prompt}", self.prompt) for arg in argv]
        else:
            argv.append(self.prompt)
        try:
            done = subprocess.run(
                argv, cwd=workdir, capture_output=True, text=True, timeout=self.timeout, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired:
            raise ProposalError(f"the agent command timed out after {self.timeout:g}s") from None
        except OSError as exc:
            raise ProposalError(f"the agent command could not start: {exc}") from None
        log = done.stdout + ("\n--- stderr ---\n" + done.stderr if done.stderr.strip() else "")
        (workdir / "proposal.txt").write_text(log, encoding="utf-8")
        manifest_path = workdir / "change_manifest.json"
        if not manifest_path.is_file():
            tail = (done.stderr or done.stdout).strip()[-300:]
            raise ProposalError(
                f"the agent command (exit {done.returncode}) did not write change_manifest.json"
                + (f": {tail}" if tail else "")
            )
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ProposalError(f"change_manifest.json is not valid JSON: {exc}") from None
        return Manifest.from_dict(data)

    @staticmethod
    def _write_feedback(ctx: Context, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.md").write_text(prompts.feedback_summary(ctx.evolve, ctx.threshold) + "\n", encoding="utf-8")
        for number, task in enumerate(ctx.evolve.task_ids()):
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", task)[:80] or "task"
            parts = [f"task: {task}"]
            for trial in ctx.evolve.for_task(task):
                parts.append(
                    f"\n=== rollout {trial.rollout}: score {trial.score:g}, "
                    f"{'passed' if trial.passed else 'failed'} ===\n{trial.trace or trial.error or '(no trace)'}"
                )
            (directory / f"{number:03d}_{safe}.txt").write_text("\n".join(parts) + "\n", encoding="utf-8")
