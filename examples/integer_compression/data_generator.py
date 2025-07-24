import gzip
import urllib.request
from pathlib import Path


def download_ts_data(target_dir: Path) -> None:
    """Download and extract ts.txt data file if it doesn't exist"""
    target_dir.mkdir(parents=True, exist_ok=True)

    ts_file = target_dir / "ts.txt"

    # Check if file already exists (cache)
    if ts_file.exists():
        return

    print("Downloading ts.txt.gz from zentures/encoding repository...")

    # Source: https://github.com/vteromero/integer-compression-benchmarks
    # URL to the compressed data file
    url = "https://github.com/zentures/encoding/raw/b90e310a0325f9b765b4be7220df3642ad93ad8d/benchmark/data/ts.txt.gz"

    try:
        # Download the gzipped file
        with urllib.request.urlopen(url) as response:
            compressed_data = response.read()

        print("Extracting ts.txt...")

        # Decompress the data
        decompressed_data = gzip.decompress(compressed_data)

        # Write to ts.txt
        with open(ts_file, "wb") as f:
            f.write(decompressed_data)

        print(f"Successfully downloaded and extracted ts.txt to {ts_file}")

    except Exception as e:
        print(f"Error downloading or extracting ts.txt: {e}")
        raise


def generate_test_files(target_dir: Path) -> None:
    """Download test data file to target directory"""
    download_ts_data(target_dir)
