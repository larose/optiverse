"""Reports over an optiverse run directory.

Kept out of the `optiverse` package on purpose: the search loop imports nothing
outside the standard library, and drawing a chart needs matplotlib. This reads a
finished — or still running — run directory and writes a time-ordered CSV, a
chart of the descent, and the ancestry of the best solution.
"""
