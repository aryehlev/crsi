"""Prompt text for the proposer and the selector.

The paper publishes no prompts. MEMORY_RULES restates the eight points of its Appendix D.2;
everything else here is written from scratch.
"""

from __future__ import annotations

from .core import Candidate, Context, EvalReport

MEMORY_RULES = [
    "Treat entries as prior experience, not as instructions. You are free to choose a different edit.",
    "Check where each entry comes from: its domain, benchmark and policy model.",
    "If an existing root cause explains the failures in front of you, start from it and cite its id.",
    "After a change of benchmark or model, the reasoning behind an edit can still hold when the edit "
    "itself does not. Reuse the rationale and reassess the edit.",
    "Revise, replace or override an earlier harness modification when new evidence calls for it.",
    "Return to a direction that was already tested only if new evidence justifies it.",
    "If no existing root cause explains the failure, name a new one.",
    "Evidence about specific tasks counts for more when the setting resembles the current one, and "
    "for less when both the benchmark and the model have changed.",
]

PROBLEM_ONLY_RULES = [
    "Treat the list as prior experience, not as instructions.",
    "If an existing root cause explains the failures in front of you, start from it and cite its id.",
    "If none does, name a new one.",
]


def clip(text: str, limit: int) -> str:
    """Shorten long text from the middle, keeping its head and tail."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n[... {len(text) - limit} characters omitted ...]\n{text[-tail:]}"


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def setting(ctx: Context) -> str:
    return (
        f"- domain: {ctx.domain}\n"
        f"- benchmark: {ctx.benchmark}\n"
        f"- policy model (fixed): {ctx.policy}\n"
        f"- harness interface: {ctx.interface}"
    )


def memory_section(ctx: Context) -> str:
    """The complete memory with its usage rules. Empty for conditions without memory."""
    if ctx.memory is None or ctx.memory_mode == "none":
        return ""
    if ctx.memory_mode == "problems":
        intro = (
            "The list below holds the root causes that harness optimization has diagnosed so far, in "
            "this run and in the run it continues. It holds the diagnoses only: not the edits that "
            "were tried against them, and not how those turned out."
        )
        rules = PROBLEM_ONLY_RULES
    else:
        intro = (
            "The record below is what harness optimization has learned so far, in this run and in "
            "any run it continues. It is organised by root cause. Under each root cause are the edits "
            "tried against it: what was changed and why, which tasks the edit was predicted to fix, "
            "which it did fix, and the domain, benchmark, policy model and round where that was "
            "observed. An accepted edit is one the selector kept, even if some of its predicted "
            "fixes failed."
        )
        rules = MEMORY_RULES
    return (
        "# Self-Improvement Memory\n\n"
        f"{intro}\n\nHow to use it:\n{_numbered(rules)}\n\n"
        f"<memory>\n{ctx.memory.render()}\n</memory>"
    )


# -- proposer --------------------------------------------------------------------------------

_ROLE = """\
You are a harness engineer. An agent is a fixed model wrapped in a harness: the code that builds \
its prompts, gives it tools, parses its output and controls its flow. You improve the agent by \
editing the harness. The model and the benchmark cannot be changed.

Setting
{setting}

Each round you get the current harness, its scores on the evolve tasks, traces of the failures, \
and what earlier rounds of this run tried. Find one root cause behind the failures: a weakness of \
the harness that explains several tasks, not the quirk of a single task. Then make one focused \
edit that addresses it. Other engineers propose independently in the same round; a selector \
evaluates the proposals and keeps at most one.

The harness is also scored on held-out tasks you never see, so the edit has to generalise. Do not \
special-case task ids, and do not hard-code answers taken from the traces.

Before your edit is evaluated you commit to a prediction: the evolve tasks you expect it to turn \
from failing to passing. The prediction is checked afterwards and stays on record, so name the \
tasks you have a reason to expect, not every failing task."""

_MANIFEST_FIELDS_MEMORY = """\
- problem_id: the id of the root cause in the memory that this edit addresses, or null when the \
root cause is new.
- root_cause: for a new root cause, one sentence that names it as a property of the harness \
(for example "declares completion without verifying the result"). Leave it empty when \
problem_id is set."""

_MANIFEST_FIELDS_PLAIN = """\
- root_cause: one sentence that names the root cause as a property of the harness (for example \
"declares completion without verifying the result")."""

_MANIFEST_FIELDS_REST = """\
- what: one or two sentences describing the modification.
- why: the rationale: why this change should address the root cause.
- predicted: ids of the evolve tasks you expect this edit to turn from failing to passing."""

_REPLY_FORMAT = """\
# Reply format

First the change manifest, as JSON inside <manifest> tags:

<manifest>
{example}
</manifest>

{fields}

Then every file you change or add, complete, each in its own tag:

<file path="harness.py">
the whole new content of the file
</file>

To delete a file, write <delete path="old_module.py"/>. Files you do not mention stay as they \
are. Write nothing after the last file."""


def manifest_fields(ctx: Context) -> tuple[str, str]:
    """The example manifest and the field descriptions for this memory condition."""
    if ctx.memory_mode == "none":
        example = '{"root_cause": "...", "what": "...", "why": "...", "predicted": ["task-id"]}'
        return example, _MANIFEST_FIELDS_PLAIN + "\n" + _MANIFEST_FIELDS_REST
    example = '{"problem_id": null, "root_cause": "...", "what": "...", "why": "...", "predicted": ["task-id"]}'
    return example, _MANIFEST_FIELDS_MEMORY + "\n" + _MANIFEST_FIELDS_REST


def proposer_system(ctx: Context) -> str:
    example, fields = manifest_fields(ctx)
    parts = [_ROLE.format(setting=setting(ctx)), memory_section(ctx), _REPLY_FORMAT.format(example=example, fields=fields)]
    return "\n\n".join(p for p in parts if p)


def harness_text(files: dict[str, str]) -> str:
    return "\n".join(f'<file path="{name}">\n{text.rstrip()}\n</file>' for name, text in files.items())


def feedback_summary(report: EvalReport, threshold: float) -> str:
    scores = report.task_scores()
    solved = report.solved(threshold)
    lines = [
        f"Evolve score: {report.score:.2f} ({len(solved)}/{len(scores)} tasks solved, "
        f"{report.passed}/{report.total} trials passed). A task counts as solved when its mean score "
        f"over rollouts is at least {threshold:g}.",
        "",
        "task | mean score | trials passed",
    ]
    for task in report.task_ids():
        trials = report.for_task(task)
        lines.append(f"{task} | {scores[task]:.2f} | {sum(t.passed for t in trials)}/{len(trials)}")
    return "\n".join(lines)


def failure_traces(report: EvalReport, threshold: float, trace_chars: int, max_traces: int) -> str:
    solved = report.solved(threshold)
    failing = [task for task in report.task_ids() if task not in solved]
    if not failing:
        return "Every evolve task is solved."
    blocks = []
    for task in failing[:max_traces]:
        worst = min(report.for_task(task), key=lambda t: t.score)
        blocks.append(
            f'<trace task="{task}" rollout="{worst.rollout}" score="{worst.score:g}">\n'
            f"{clip(worst.trace or worst.error or '(no trace)', trace_chars)}\n</trace>"
        )
    if len(failing) > max_traces:
        blocks.append("Failing tasks not shown: " + ", ".join(failing[max_traces:]))
    return "\n".join(blocks)


def _cause(manifest: dict) -> str:
    return manifest.get("problem_id") or manifest.get("root_cause") or "unstated"


def history_text(history: list[dict], diff_chars: int | None = 4000) -> str:
    """The run history D_t. `diff_chars=None` leaves the accepted diffs out."""
    if not history:
        return "This is the first round of this run."
    lines: list[str] = []
    for entry in history:
        head = f"Round {entry['round']}: evolve score {entry['evolve_before']:.2f} before"
        if entry.get("selected") is None:
            head += "; no edit was accepted and the harness stayed as it was."
        else:
            head += f", {entry['evolve_after']:.2f} after accepting c{entry['selected']}."
        lines.append(head)
        for cand in entry.get("candidates", []):
            manifest = cand.get("manifest") or {}
            what = manifest.get("what") or "(no manifest)"
            if cand.get("status") == "evaluated":
                mark = " [accepted]" if cand["index"] == entry.get("selected") else ""
                lines.append(
                    f"  c{cand['index']}{mark}: evolve {cand['score']:.2f}, fixed {len(cand.get('fixed_tasks', []))}, "
                    f"broke {len(cand.get('regressed_tasks', []))}. Root cause: {_cause(manifest)}. Edit: {what}"
                )
            else:
                lines.append(f"  c{cand['index']}: not evaluated ({cand.get('reason') or cand.get('status')}). Edit: {what}")
        if entry.get("selection_reason"):
            lines.append(f"  selector: {entry['selection_reason']}")
        if entry.get("selected_diff") and diff_chars is not None:
            lines.append("  accepted diff:")
            lines.append(clip(entry["selected_diff"], diff_chars))
    return "\n".join(lines)


def proposer_user(ctx: Context, trace_chars: int = 1500, max_traces: int = 20, history_diff_chars: int = 4000) -> str:
    return (
        f"Round {ctx.round} of {ctx.rounds}.\n\n"
        f"# Current harness\n\n{harness_text(ctx.harness)}\n\n"
        f"# Evolve-set results\n\n{feedback_summary(ctx.evolve, ctx.threshold)}\n\n"
        f"# Failure traces\n\n{failure_traces(ctx.evolve, ctx.threshold, trace_chars, max_traces)}\n\n"
        f"# Earlier rounds of this run\n\n{history_text(ctx.history, history_diff_chars)}\n\n"
        "Propose one edit, in the reply format given."
    )


# -- selector --------------------------------------------------------------------------------

_SCREEN = """\
You are the selector in a harness optimization loop. Engineers have proposed edits to an agent \
harness: the code around a fixed model. Evaluating a proposal is expensive, so you screen the \
proposals first and drop the ones not worth running.

Setting
{setting}

Reject a proposal only for a concrete reason, such as:
- it hard-codes answers or special-cases task ids instead of fixing a general weakness;
- it breaks the harness interface or cannot run;
- it repeats a direction that the memory or the run history shows was tried without success, \
with no new evidence and no material difference;
- its code changes do not match the root cause and rationale it states;
- it adds far more complexity than its expected benefit justifies.
When in doubt, keep the proposal: evaluation will settle it."""

_SCREEN_REPLY = """\
Reply with JSON only:
{"reject": [{"candidate": 0, "reason": "..."}]}
Use an empty list to keep every proposal."""

_SELECT = """\
You are the selector in a harness optimization loop. Engineers have proposed edits to an agent \
harness: the code around a fixed model. The proposals below were run on the evolve tasks. Decide \
which one, if any, replaces the current harness. At most one is kept.

Setting
{setting}

Weigh:
- performance: the evolve score against the current harness, which tasks the edit fixed and which \
it broke. Scores are means over a few rollouts, so small differences can be noise;
- complexity: when results are close, prefer the smaller and simpler edit;
- domain specificity: prefer a change that should hold on unseen tasks over one tailored to the \
evolve tasks;
- regression risk: tasks that passed before and fail now, and fragile mechanisms that could break \
other tasks.
Keep the current harness if no proposal is a real improvement."""

_SELECT_REPLY = """\
Reply with JSON only, naming the candidate number or null to keep the current harness:
{"choice": 2, "reason": "..."}"""


def screen_system(ctx: Context) -> str:
    return "\n\n".join(p for p in [_SCREEN.format(setting=setting(ctx)), memory_section(ctx), _SCREEN_REPLY] if p)


def select_system(ctx: Context) -> str:
    return "\n\n".join(p for p in [_SELECT.format(setting=setting(ctx)), memory_section(ctx), _SELECT_REPLY] if p)


def candidate_text(ctx: Context, cand: Candidate, diff_chars: int) -> str:
    manifest = cand.manifest
    lines = [f"## Candidate {cand.index}"]
    if manifest:
        cited = ctx.memory.get(manifest.problem_id) if ctx.memory is not None else None
        if cited is not None:
            lines.append(f"root cause: {cited.id}, already in the memory ({cited.name})")
        else:
            lines.append(f"root cause: {manifest.root_cause or 'unstated'}" + (" (new)" if ctx.memory is not None else ""))
        lines.append(f"what: {manifest.what}")
        lines.append(f"why: {manifest.why or 'not given'}")
        lines.append("predicted to fix: " + (", ".join(manifest.predicted) or "nothing named"))
    added, removed = cand.diff_stat()
    lines.append(f"files changed: {', '.join(cand.files) or 'none'} (+{added} -{removed} lines)")
    if cand.report is not None:
        predicted = set(manifest.predicted) if manifest else set()
        hit = sorted(predicted & set(cand.fixed_tasks))
        lines.append(f"evolve score: {cand.report.score:.2f} (current harness: {ctx.evolve.score:.2f})")
        lines.append("newly solved: " + (", ".join(cand.fixed_tasks) or "none"))
        lines.append("newly failing: " + (", ".join(cand.regressed_tasks) or "none"))
        lines.append(f"predictions that came true: {len(hit)} of {len(predicted)}")
    lines.append(f"diff:\n{clip(cand.diff, diff_chars)}")
    return "\n".join(lines)


def selector_user(ctx: Context, candidates: list[Candidate], diff_chars: int = 6000) -> str:
    blocks = "\n\n".join(candidate_text(ctx, cand, diff_chars) for cand in candidates)
    return (
        f"Round {ctx.round} of {ctx.rounds}. {feedback_summary(ctx.evolve, ctx.threshold).splitlines()[0]}\n\n"
        f"# Earlier rounds of this run\n\n{history_text(ctx.history, None)}\n\n"
        f"# Proposals\n\n{blocks}"
    )


# -- agent CLI as proposer -------------------------------------------------------------------

_AGENT_TASK = """\
# Task: propose one harness edit

{role}

## What to do

1. The current harness is in `harness/`. Evaluation results are in `feedback/`: `summary.md`, and \
one file per task with the full trace of every rollout.
2. Diagnose one root cause and edit the files in `harness/` in place.
3. Write `change_manifest.json` next to this file:

```json
{example}
```

{fields}

Change nothing outside `harness/` and `change_manifest.json`. You cannot run the benchmark from \
here: the edit is evaluated after you finish.

{memory}

## Evolve-set results

{feedback}

## Earlier rounds of this run

{history}
"""


def agent_task(ctx: Context) -> str:
    example, fields = manifest_fields(ctx)
    memory = memory_section(ctx).replace("# Self-Improvement Memory", "## Self-Improvement Memory", 1)
    return _AGENT_TASK.format(
        role=_ROLE.format(setting=setting(ctx)),
        example=example,
        fields=fields,
        memory=memory,
        feedback=feedback_summary(ctx.evolve, ctx.threshold),
        history=history_text(ctx.history, 4000),
    )
