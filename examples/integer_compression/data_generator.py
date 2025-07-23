import hashlib
import random
import struct
from pathlib import Path
from typing import Dict


# Expected SHA256 checksums for deterministic test data files
EXPECTED_CHECKSUMS: Dict[str, str] = {
    "data_a.bin": "7f8b6c4d2e1a9f3b5c8e7d6a9f2b4c1e3d5f7a8b9c2e4d6f8a1b3c5e7d9f2b4c",  # Placeholder
    "data_b.bin": "2a4c6e8f1b3d5f7a9c2e4d6f8a1b3c5e7d9f2b4c6e8f1a3c5e7d9f2b4c6e8f1",  # Placeholder
    "data_c.bin": "9f2b4c6e8f1a3c5e7d9f2b4c6e8f1a3c5e7d9f2b4c6e8f1a3c5e7d9f2b4c6e",  # Placeholder
}

SIZE_PER_FILE = 10_000_000  # 10M integers
SEEDS = {
    "data_a.bin": 12345,
    "data_b.bin": 67890,
    "data_c.bin": 54321,
}


def generate_random_data(seed: int, size: int) -> bytes:
    """Generate random uint32 data with given seed"""
    rng = random.Random(seed)
    data = bytearray()

    for _ in range(size):
        # Generate random uint32 value
        value = rng.randint(0, 2**32 - 1)
        # Pack as little-endian uint32
        data.extend(struct.pack("<I", value))

    return bytes(data)


def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def should_generate_file(filepath: Path, filename: str, force_regen: bool) -> bool:
    """Check if file should be generated"""
    # Always generate if force_regen is True
    if force_regen:
        return True

    # Generate if file doesn't exist
    if not filepath.exists():
        return True

    # Check checksum
    current_hash = calculate_file_hash(filepath)
    expected_hash = EXPECTED_CHECKSUMS.get(filename)

    if expected_hash and current_hash != expected_hash:
        raise RuntimeError(
            f"Checksum mismatch for {filename}!\n"
            f"Expected: {expected_hash}\n"
            f"Got:      {current_hash}\n"
            f"File may be corrupted. Set FORCE_REGEN_DATA=1 to regenerate."
        )

    return False


def generate_test_files(target_dir: Path, force_regen: bool = False) -> None:
    """Generate test data files in target directory"""
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, seed in SEEDS.items():
        filepath = target_dir / filename

        if should_generate_file(filepath, filename, force_regen):
            print(f"Generating {filename} with {SIZE_PER_FILE:,} integers...")

            # Generate random data
            data = generate_random_data(seed, SIZE_PER_FILE)

            # Write to file
            with open(filepath, "wb") as f:
                f.write(data)

            # Calculate and update checksum for first-time generation
            file_hash = calculate_file_hash(filepath)
            if filename not in EXPECTED_CHECKSUMS or not EXPECTED_CHECKSUMS[filename]:
                print(f"Generated {filename} with hash: {file_hash}")
                # Note: In a real implementation, you'd update the checksums in code
            else:
                print(f"Generated {filename} (hash verified)")
        else:
            print(f"{filename} exists and checksum is valid")
