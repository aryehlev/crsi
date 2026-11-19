from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import FakeModelServer, ScriptedProposer, config, make_source, step  # noqa: E402
from toybench import SEED, FakePolicy, ToyBench  # noqa: E402

from crsi import GreedySelector, continue_rsi  # noqa: E402


@pytest.fixture
def seed_harness(tmp_path: Path) -> Path:
    path = tmp_path / "seed"
    path.mkdir()
    (path / "harness.py").write_text(SEED)
    return path


@pytest.fixture
def source_run(tmp_path: Path, seed_harness: Path) -> Path:
    make_source(tmp_path, seed_harness)
    return tmp_path / "src"


@pytest.fixture
def continued_run(tmp_path: Path, source_run: Path) -> Path:
    """Full SIM on toy-B, continuing `source_run`. This is the run behind the README's example."""
    plan = {
        (1, 0): step(
            "add a reverse handler", ("strip", "reverse"), ["reverse-0", "reverse-1"], problem_id="P1",
            why="attempt 1 predicted every kind and fixed only strip, so this edit names just the reverse tasks it handles",
        )
    }
    bench = ToyBench(FakePolicy(), name="toy-B", kinds=("strip", "reverse", "lower"))
    continue_rsi(bench, "full_sim", ScriptedProposer(plan), GreedySelector(), config(tmp_path, "tgt", candidates=1), source_dir=source_run)
    return tmp_path / "tgt"


@pytest.fixture
def model_server(monkeypatch):
    server = FakeModelServer()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", server.url + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield server
    server.close()
