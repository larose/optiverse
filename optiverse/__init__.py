"""Optiverse: evolve codebases with coding agents.

One concept per folder. `optiverse.search` is what the run knows and how the
next move is chosen; `optiverse.director` and `optiverse.programmer` are the two
agents that drive it; `optiverse.solution` and `optiverse.evaluator` are what a
candidate is and what scores it.

Everything imported here is standard library only. The two agent
implementations — `optiverse.director.agent` and `optiverse.programmer.agent` —
are not, and are deliberately left out, along with `optiverse.mini_swe_agent`
underneath them: importing any of the three pulls mini-swe-agent and its
dependency tree, so `import optiverse` costs nothing until you ask by name.
"""

from . import config
from . import director
from . import evaluator
from . import optimizer
from . import programmer
from . import search
from . import solution

__all__ = [
    "config",
    "director",
    "evaluator",
    "optimizer",
    "programmer",
    "search",
    "solution",
]
