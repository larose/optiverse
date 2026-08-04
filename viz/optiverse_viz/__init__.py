"""A browser view of an optiverse run's node graph.

    python -m optiverse_viz <run directory>

Writes `<run>/viz/index.html`: the tree of ideas with every constraint on it, and
the code lineage drawn over the top. One self-contained file — the vendored d3,
the page's own script and style, and the run's data are all inlined, so it opens
offline from any path and can be copied somewhere else on its own.

This is a separate distribution from `optiverse` and depends on it rather than
the other way round. The loop ships with no dependency it does not need to run,
and a report is not something a run needs.
"""

from .model import build
from .render import render, write

__all__ = ["build", "render", "write"]
