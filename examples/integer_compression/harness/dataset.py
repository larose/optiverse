#!/usr/bin/env python3
"""The benchmark dataset: fetch it, and convert it once into a form worth reading.

Run once before scoring:

    python examples/integer_compression/harness/dataset.py

Two files land here, both gitignored:

    ts.txt   ~1.6 GB of decimal integers, one per line, as published
    ts.bin   ~577 MB of raw little-endian uint32, the same values

`ts.bin` exists because `ts.txt` is text. Parsing 144 million integers takes tens
of seconds, and scoring a candidate runs the harness several times over — paying
that on every run made scoring minutes long and drowned the thing being measured.
Converting once turns each run's load into a sequential read of half a gigabyte.

Validation needs neither file, so an agent's inner loop never touches this.
"""

import gzip
import struct
import sys
import urllib.request
from pathlib import Path
from typing import List

# Source: https://github.com/vteromero/integer-compression-benchmarks
DATA_URL = (
    "https://github.com/zentures/encoding/raw/"
    "b90e310a0325f9b765b4be7220df3642ad93ad8d/benchmark/data/ts.txt.gz"
)

HARNESS_DIRECTORY = Path(__file__).parent
TEXT_FILE = HARNESS_DIRECTORY / "ts.txt"
BINARY_FILE = HARNESS_DIRECTORY / "ts.bin"

# Values per write. Large enough that the packing is not the bottleneck, small
# enough that the whole file is never held in memory.
CHUNK_VALUES = 1 << 20


def download(target: Path = TEXT_FILE) -> None:
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


def convert(source: Path = TEXT_FILE, target: Path = BINARY_FILE) -> None:
    """Rewrite the decimal text as raw little-endian uint32.

    Also the only place the data is held to what `problem.md` promises a
    candidate: non-decreasing, and inside the range of a uint32.
    """
    if target.exists():
        print(f"{target} already exists")
        return

    print(f"Converting {source} to {target}...")

    count = 0
    previous = -1
    chunk: List[int] = []

    # Written beside the target and renamed at the end, so an interrupted
    # conversion cannot leave a half-written file that looks finished.
    temporary = target.with_suffix(".bin.tmp")

    with source.open() as text, temporary.open("wb") as binary:
        for line in text:
            stripped = line.strip()

            if not stripped:
                continue

            value = int(stripped)

            if not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(f"{source}: {value} does not fit in a uint32")

            if value < previous:
                raise ValueError(
                    f"{source}: {value} follows {previous}; the data must be "
                    "non-decreasing"
                )

            previous = value
            chunk.append(value)
            count += 1

            if len(chunk) == CHUNK_VALUES:
                binary.write(struct.pack(f"<{len(chunk)}I", *chunk))
                chunk.clear()

        if chunk:
            binary.write(struct.pack(f"<{len(chunk)}I", *chunk))

    temporary.replace(target)

    print(f"Wrote {target} ({count} values, {target.stat().st_size} bytes)")


def main() -> None:
    download()
    convert()


if __name__ == "__main__":
    main()
    sys.exit(0)
