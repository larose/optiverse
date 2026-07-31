"""Optiverse: evolve codebases with coding agents.

The core imports nothing outside the standard library. The two things that drive
a coding agent — `optiverse.generators`, which writes candidates, and
`optiverse.strategists`, which decides what to try next — are imported explicitly.
"""

from . import codebase
from . import config
from . import evaluator
from . import evaluator_main
from . import generator
from . import optimizer
from . import prompt_generator
from . import search
from . import store
from . import strategist

__all__ = [
    "codebase",
    "config",
    "evaluator",
    "evaluator_main",
    "generator",
    "optimizer",
    "prompt_generator",
    "search",
    "store",
    "strategist",
]
