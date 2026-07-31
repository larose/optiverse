# Analysis

Reports over a run directory: what the search produced, and how its best
solution came to exist.

This is a separate distribution with its own virtualenv. `optiverse` imports
nothing outside the standard library and drawing a chart needs matplotlib, so
the dependency lives here and the search loop stays free of it. Nothing in
`optiverse` imports this.

## Use

```bash
cd analysis
make init
make analyze DIRECTORY=../tmp/20260730_141954
```

Three files land in `<run>/analysis/`, beside the run they describe:

- `solutions_timeline.csv` — every solution in the order it was committed, with
  the running best next to it. The run's own `solutions.csv` is sorted by score
  and has no clock; this is the other view. Untouched by this tool.
- `lineage_tree.png` — the family tree. The initial solution is the root, each
  child hangs below its parent, colour is the score, and the thread down to the
  best solution is highlighted.
- `lineage.md` — that thread written out: every step, what it scored, what the
  agent said it did, and the diff against its parent.

A short version goes to the terminal.

## Timestamps

Optiverse records no per-solution timestamp and no iteration index. The mtime of
`<id>/metadata.json` — written atomically and last — is the only record of when
a solution was committed, and everything ordered here rests on it.

So copying a run directory needs `cp -a`, `rsync -a` or `tar`. Plain `cp -r`
stamps every solution with the moment of the copy and the history collapses to a
point. When that has happened the tool says so rather than drawing a chart with
a meaningless axis, but it cannot recover what was lost — re-copy from the
original.

## Development

```bash
make format
make test          # black, pyright, unittest
```

`make test` at the repository root covers the core and does not build this
virtualenv; run this one from here.
