import random
import struct
from pathlib import Path


SIZE_PER_FILE = 10_000_000
SEEDS = {
    "data_a.bin": 12345,
    "data_b.bin": 67890,
    "data_c.bin": 54321,
}


def generate_random_data(seed: int, size: int) -> bytes:
    """Generate sorted uint32 data with given seed"""
    rng = random.Random(seed)

    # Generate random integers efficiently
    values = [rng.randint(0, 2**32 - 1) for _ in range(size)]

    # Sort to create sorted integer sequence
    values.sort()

    # Pack all values at once using batch packing for efficiency
    return struct.pack(f"<{size}I", *values)


def generate_test_files(target_dir: Path) -> None:
    """Generate test data files in target directory"""
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, seed in SEEDS.items():
        filepath = target_dir / filename

        if filepath.exists():
            continue

        print(f"Generating {filename} with {SIZE_PER_FILE:,} integers...")

        # Generate random data
        data = generate_random_data(seed, SIZE_PER_FILE)

        # Write to file
        with open(filepath, "wb") as f:
            f.write(data)
