"""Benchmarks.

A benchmark is any object with

    name, domain, policy      strings: the benchmark, its domain, the policy model's name
    interface                 text shown to the proposer: what a harness must provide
    tasks(split)              task ids for "evolve" or "heldout"
    run(harness_dir, task_id, rollout) -> TaskResult

and optionally `validate(harness_dir) -> str | None`, a cheap check run before evaluation that
returns a reason to reject a harness.
"""

from __future__ import annotations

import importlib
import os
import sys

from . import generated

BUILTIN = {"arithmetic": generated.arithmetic, "strings": generated.strings}


def load_benchmark(spec: str, policy, **kwargs):
    """Build a benchmark from a built-in name or from "module:factory", called as factory(policy, **kwargs)."""
    if spec in BUILTIN:
        return BUILTIN[spec](policy, **kwargs)
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"unknown benchmark {spec!r}: use one of {sorted(BUILTIN)} or 'module:factory'")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(module_name), attr)(policy, **kwargs)
