#!/usr/bin/env python3
"""Fetch the benchmark dataset used for scoring.

Run once before scoring:

    python examples/integer_compression/data_generator.py

The file is ~1.6 GB and gitignored. Validation does not need it, so an agent's
inner loop never touches it.
"""

import gzip
import sys
import urllib.request
from pathlib import Path

# Source: https://github.com/vteromero/integer-compression-benchmarks
DATA_URL = (
    "https://github.com/zentures/encoding/raw/"
    "b90e310a0325f9b765b4be7220df3642ad93ad8d/benchmark/data/ts.txt.gz"
)

DEFAULT_TARGET = Path(__file__).parent / "harness" / "ts.txt"


def download_ts_data(target: Path = DEFAULT_TARGET) -> None:
    if target.exists():
        print(f"{target} already exists")
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATA_URL}...")
    with urllib.request.urlopen(DATA_URL) as response:
        compressed_data = response.read()

    print("Extracting...")
    target.write_bytes(gzip.decompress(compressed_data))

    print(f"Wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    download_ts_data()
    sys.exit(0)
