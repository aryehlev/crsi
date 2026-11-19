"""The README makes claims about output and commands. These tests keep them true."""

import re
import shlex
from pathlib import Path

from crsi import SIM
from crsi.cli import build_parser

README = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")


def test_the_memory_example_is_what_the_fixture_run_really_renders(continued_run):
    assert SIM.load(continued_run / "sim.json").render() in README


def test_the_quickstart_commands_are_accepted_by_the_cli():
    block = re.search(r"## Quickstart\n\n```bash\n(.*?)```", README, re.S).group(1)
    variables = dict(re.findall(r"^(\w+)=\"?([^\"#\n]+?)\"?\s*(?:#.*)?$", block, re.M))
    commands = re.findall(r"^\s*(crsi (?:[^\\\n]|\\\n)*)", block, re.M)
    assert len(commands) == 4
    parser = build_parser()
    seen = []
    for command in commands:
        command = command.replace("\\\n", " ").replace("$arm", "full_sim")
        for name, value in variables.items():
            command = command.replace(f"${name}", value)
        assert "$" not in command, command
        args = parser.parse_args(shlex.split(command)[1:])
        seen.append(args.command)
        if args.command in ("run", "continue"):
            assert (args.rounds, args.candidates, args.rollouts, args.bench_arg) == (3, 3, 2, ["n=24"])
    assert seen == ["run", "continue", "report", "show"]


def test_the_documented_defaults_are_the_papers_budget():
    args = build_parser().parse_args(["run", "--bench", "arithmetic", "--harness", "h", "--out", "o", "--policy", "p:m"])
    assert (args.rounds, args.candidates, args.rollouts) == (10, 5, 5)
    assert "10 rounds, 5 candidates per round, 5 rollouts" in README
