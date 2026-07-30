"""Optiverse: evolve codebases with coding agents.

The core imports nothing outside the standard library. Generators that need a
third-party agent live in `optiverse.generators` and are imported explicitly.
"""

from . import codebase
from . import config
from . import evaluator
from . import evaluator_main
from . import generator
from . import optimizer
from . import prompt_generator
from . import search_strategies
from . import store

__all__ = [
    "codebase",
    "config",
    "evaluator",
    "evaluator_main",
    "generator",
    "optimizer",
    "prompt_generator",
    "search_strategies",
    "store",
]
