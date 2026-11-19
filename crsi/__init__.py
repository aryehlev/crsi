"""crsi: an unofficial implementation of Self-Improvement Memory (SIM), arXiv 2611.18856."""

__version__ = "0.1.0"

from .continual import ARMS, continue_rsi, run_source
from .core import Candidate, Context, EvalReport, Manifest, ProposalError, RunError, TaskResult
from .evaluate import evaluate
from .harness import load_harness
from .llm import LLMError, LLMRefusal, make_llm
from .loop import RunConfig, optimize
from .memory import SIM, Attempt, Problem
from .proposer import CommandProposer, LLMProposer
from .selector import GreedySelector, LLMSelector
from .stats import paired

__all__ = [
    "ARMS", "SIM", "Attempt", "Candidate", "CommandProposer", "Context", "EvalReport", "GreedySelector",
    "LLMError", "LLMProposer", "LLMRefusal", "LLMSelector", "Manifest", "Problem", "ProposalError",
    "RunConfig", "RunError", "TaskResult", "continue_rsi", "evaluate", "load_harness", "make_llm",
    "optimize", "paired", "run_source",
]
