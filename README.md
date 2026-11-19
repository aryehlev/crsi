# crsi

An unofficial implementation of **Self-Improvement Memory (SIM)** from
[*Self-Improvement Memory Enables Continual Recursive Self-Improvement*](https://arxiv.org/abs/2611.18856)
(Yu et al., Google, November 2026).

Harness optimizers improve an agent by rewriting the code around a fixed model: its prompts,
tools and control flow. When the benchmark or the model changes, the evolved harness carries
over, but the reasons behind its edits do not. SIM is a memory for the *optimizer*: every
accepted edit is filed under the root cause it addresses, with its rationale, the tasks it
was predicted to fix, the tasks it did fix, and where that was observed. Continued
optimization reads and extends that record.

## Status

- **Unofficial.** Not affiliated with the authors. The paper had no code release when this was written (2026-11-19).
- **Not yet run against a real model.** The 83 tests use scripted stand-ins for the optimizer and a local fake model server. They verify the loop, the memory and the four continuation arms. They do not verify that the prompts work well, and **none of the paper's results have been reproduced here.**
- The paper's benchmarks are not bundled. Two small generated benchmarks are included for trying the loop.

## Install

```bash
pip install -e .        # Python 3.10+, no dependencies
pytest                  # optional: pip install pytest
```

## Quickstart

```bash
export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY, plus OPENAI_BASE_URL for other servers
POLICY=anthropic:claude-haiku-5-5     # the fixed model that solves tasks
OPT=anthropic:claude-sonnet-5-5       # the model that proposes and selects harness edits
SMALL="--rounds 3 --candidates 3 --rollouts 2 --bench-arg n=24"   # drop this for the paper's budget

# 1. Source RSI: evolve the seed harness on one benchmark, building a memory.
crsi run --bench arithmetic --harness examples/seed_harness --out runs/src \
     --policy $POLICY --optimizer $OPT $SMALL

# 2. Continue on a different benchmark, under each of the paper's four conditions.
for arm in cold_start harness_transfer problem_only full_sim; do
  crsi continue --bench strings --from runs/src --arm $arm --seed-harness examples/seed_harness \
       --out runs/$arm --policy $POLICY --optimizer $OPT $SMALL
done

# 3. Compare held-out scores, and read the memory the way the optimizer sees it.
crsi report runs/cold_start runs/harness_transfer runs/problem_only runs/full_sim
crsi show runs/full_sim
```

With several independent source runs, `crsi paired` runs the paper's significance test on the
final held-out scores, pairing the two conditions by source run:

```bash
crsi paired --treatment runs/s1/full_sim runs/s2/full_sim runs/s3/full_sim \
            --control   runs/s1/harness_transfer runs/s2/harness_transfer runs/s3/harness_transfer
```

For cross-model continuation, keep `--bench` and change `--policy`. A finished `full_sim` run
is a valid `--from`, so generations can be chained. Runs resume: repeat the same command and
it continues after the last completed round.

**Budget.** The defaults are the paper's: 10 rounds, 5 candidates per round, 5 rollouts per
task. With the default 60-task generated benchmark that is about 8,000 task runs per run, each
at least one policy call. The `SMALL` settings above come to about 300.

**Safety.** A harness is code written by a model, and it is imported and executed in the
optimizer's process. Run this in a container or VM.

## What a memory looks like

`crsi show` on a continued run. This one is from the scripted test fixture, not a real model: the
source run over-predicted (six tasks named, two fixed), and the continued run can see that.

```
Carried from run 'src' (domain: toy) into run 'tgt'.

## P1: the harness returns the input unchanged for task kinds it has no handler for
first diagnosed in round 1; 2 attempt(s)
- attempt 1: round 1, sample 1 (carried from run 'src'); accepted; domain=toy, benchmark=toy-A, policy=model-1
  what: add a strip handler
  why: strip tasks come back with their padding
  files: harness.py
  predicted to fix (6): reverse-0, reverse-1, strip-0, strip-1, upper-0, upper-1
  observed: fixed (2): strip-0, strip-1; missed (4): reverse-0, reverse-1, upper-0, upper-1; evolve trials passed 2/6
- attempt 2: round 1, sample 0; accepted; domain=toy, benchmark=toy-B, policy=model-1
  what: add a reverse handler
  why: attempt 1 predicted every kind and fixed only strip, so this edit names just the reverse tasks it handles
  files: harness.py
  predicted to fix (2): reverse-0, reverse-1
  observed: fixed (2): reverse-0, reverse-1; missed (0); evolve trials passed 4/6
```

On disk it is `sim.json`, using the field names of the paper's Appendix D.3: `next_id`,
`carried_from`, `aliases`, `problems[].{id, name, opened_round, attempts[]}` and per attempt
`round, sample, accepted, backfilled, what, files, predicted, fixed, missed, passed, total, domain`.

## How it maps to the paper

| Paper | Here |
| --- | --- |
| Memory entry `m_i = (c_i, [(a_ij, o_ij, p_ij)])`, section 3.2 | `crsi/memory.py`: `SIM`, `Problem`, `Attempt` |
| Update `U(M_t, e_t)`: append to the cited root cause or open a new one | `SIM.record` |
| Proposer decides the grouping; identical root-cause text is merged, histories kept | manifest `problem_id` / `root_cause`; `SIM.merge_identical`, `aliases` |
| Proposer `P(H_t, F_t, D_t, M_t)`, k candidates, full memory in the system prompt, no retrieval or truncation | `crsi/proposer.py`, `SIM.render` |
| Change manifest written before evaluation | `rounds/rNN/cK/manifest.json` |
| Selector screens before execution, then chooses by performance, complexity, domain specificity and regression risk, reading the same memory | `crsi/selector.py` |
| Only the selected edit is recorded, even if it regresses; no acceptance leaves harness and memory unchanged | `crsi/loop.py` (`record="selected"`) |
| Optimizer sees evolve-set results only; held-out is scored before and after | `crsi/loop.py`, `Context` |
| Cold Start, Harness Transfer, Problem-Only Memory, Full SIM | `crsi/continual.py` |
| Two-sided paired t test with its interval, signed-rank check and effect size, section 4.4 | `crsi/stats.py`, `crsi paired` (the signed-rank p-value is computed exactly) |

## Where the paper is silent

These are this implementation's choices. Each is a place where a faithful reproduction might differ.

- **Prompts.** The paper publishes none. The eight memory-use rules in `crsi/prompts.py` restate its Appendix D.2; the rest is written from scratch.
- **Selector.** No scoring rule or screening checks are given. Here, screening is static checks (unparseable, empty, duplicate, syntax error, a manifest that names no root cause the memory has or can add, benchmark `validate` hook) plus one model call; selection is one model call, falling back to the best evolve score if the reply is unreadable. `--greedy` skips the model.
- **Fixed and missed.** A task counts as solved when its mean score over rollouts reaches 0.5 (`RunConfig.threshold`). A predicted task is `fixed` if it went from unsolved to solved, otherwise `missed`.
- **Provenance.** The text says provenance covers domain, benchmark, policy and round, but the JSON shown has only `domain` and `round`. Attempts here also store `benchmark`, `policy`, `run` and `why`.
- **`backfilled`.** Listed in the schema, never defined. It is stored and always `false`.
- **`passed` and `total`.** In the schema, not defined. Here they count the evolve trials (task x rollout) the edited harness passed, out of all evolve trials.
- **Rendering.** How the memory is laid out in the prompt is not specified. The built-in proposer shows the whole memory, but shortens long failure traces and shows at most 20 failing tasks per round (`LLMProposer(trace_chars=..., max_traces=...)`).
- **Sampling.** The k candidates are independent calls that do not see each other.
- **Problem-Only Memory during the target run.** It keeps gaining root causes, never attempts.
- **Root-cause identity.** "Identical text" is compared ignoring case and whitespace.

`record="evaluated"` is an extension: it also logs evaluated candidates the selector rejected.

## Your own benchmark

```python
from crsi import TaskResult, load_harness

class MyBench:
    name = "my-bench-v1"
    domain = "coding"
    interface = "harness.py must define run(task, llm) -> str"   # shown to the proposer

    def __init__(self, policy):                 # policy: an object with .name and .complete(system, user)
        self.llm, self.policy = policy, policy.name

    def tasks(self, split):                     # "evolve" or "heldout"
        ...

    def run(self, harness_dir, task_id, rollout) -> TaskResult:
        answer = load_harness(harness_dir).run(self.get(task_id), self.llm)
        ok = self.grade(task_id, answer)
        return TaskResult(task_id, rollout, float(ok), ok, trace="what the optimizer should read")

def factory(policy, **kwargs):
    return MyBench(policy, **kwargs)
```

```bash
crsi run --bench my_module:factory --harness my_seed/ --out runs/src --policy ... --optimizer ...
```

A benchmark may run the harness however it likes (subprocess, container); `load_harness` is a
convenience for in-process harnesses.

## A coding agent as the proposer

The built-in proposer makes one model call with the whole harness inlined, which suits small
harnesses. For larger ones, hand the job to an agent CLI. It gets a scratch directory with
`TASK.md` (instructions, memory, feedback, history), a copy of the harness to edit in place,
full traces under `feedback/`, and must write `change_manifest.json`. Any agent CLI works
(Claude Code, Codex, Gemini CLI, OpenCode): the instruction is appended as the last argument,
or put where the command says `{prompt}`.

```bash
crsi run ... --proposer-cmd "claude -p --permission-mode acceptEdits" --selector-model $OPT
```

This path is tested with a stub script only.

## Python API

```python
from crsi import (SIM, LLMProposer, LLMSelector, RunConfig, continue_rsi, make_llm, run_source)
from crsi.benchmarks.generated import arithmetic, strings

policy, opt = make_llm("anthropic:claude-haiku-5-5"), make_llm("anthropic:claude-sonnet-5-5")
agents = (LLMProposer(opt), LLMSelector(opt))

run_source(arithmetic(policy), "examples/seed_harness", *agents, RunConfig(name="src", out="runs/src"))
result = continue_rsi(strings(policy), "full_sim", *agents, RunConfig(name="tgt", out="runs/tgt"),
                      source_dir="runs/src")
print(result["heldout_before"], result["heldout_after"])
print(SIM.load("runs/tgt/sim.json").render())
```

## Run directory

```
runs/tgt/
  config.json  state.json  result.json
  sim.json                     latest memory
  retained/                    latest retained harness
  heldout_before.json  heldout_after.json
  start/                       starting harness, its evolve report, starting memory
  rounds/r01/
    round.json                 what was proposed, screened, scored and chosen
    sim.json                   memory after this round
    c0/ manifest.json  proposal.txt  edit.diff  harness/  eval.json
```

## Limitations

- Task timeouts abandon the worker thread rather than killing it.
- No token or cost accounting.
- The generated benchmarks are toys. Conclusions about SIM need real benchmarks and repeated runs: the paper's own significance test uses ten paired source runs.

## Citation

```bibtex
@article{yu2026sim,
  title   = {Self-Improvement Memory Enables Continual Recursive Self-Improvement},
  author  = {Yu, Xin and Xia, Fei and Pan, Jiaqi and Tang, Youbao and Liu, Chengzhi and Yu, Adams and
             Jiang, Sicong and Chen, Yanfei and Xue, Lingzhou and Yu, Ting},
  journal = {arXiv preprint arXiv:2611.18856},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
