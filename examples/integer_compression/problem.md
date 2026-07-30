Design and implement a compression algorithm for sorted 32-bit unsigned integers in Go. Your goal is the lowest possible decompression time.

To be competitive, your solution must surpass traditional methods like Variable Byte (VByte), PForDelta, and VTEnc. Drawing inspiration from these or related techniques is allowed, but mere reimplementation is insufficient. True innovation is required.

You are encouraged to explore novel approaches based on well-known general patterns, including but not limited to:

- Delta Encoding and Delta-of-Delta
- Bit-Packing and Frame-of-Reference
- Run-Length Encoding
- Headerless or Self-Describing Formats
- Table-Driven and SIMD-Accelerated Decoding
- Block-based or chunked compression with skippable blocks

## Your codebase

Your codebase is a directory. Every `.go` file in it belongs to `package candidate`, and together they must define:

```go
func Compress(data []uint32) []byte
func Decompress(compressed []byte) []uint32
```

You may split the package across as many files as you like; add, rename or delete them freely. Define every function at the package level; do not nest functions.

Do not add `go.mod` or `go.sum`. The Go module belongs to the harness, and one of your own would take your code out of the build.

## How your code is run

Your package is compiled into a benchmark program that imports it, loads a dataset you never see, and calls `Compress` once and then `Decompress` once on what came back. `Decompress(Compress(data))` must return exactly `data`.

Each run is a **fresh process**. Nothing survives from one to the next: no package-level state, no cached tables, no files. Anything `Decompress` needs must be built during that run or carried in the bytes `Compress` produced.

You cannot build, run or time your code. `validate` is the only thing you can execute: it answers valid or invalid and prints diagnostics. It never tells you a decompression time, a compression ratio, or a score.

## Requirements

Use only the Go Standard Library. Third-party packages are not allowed, and the build has module downloads disabled, so an import of anything else fails to compile.

The built-in compression libraries (`compress/gzip`, `flate`, `zlib`, `lzw`, and the rest) are strictly prohibited.

Do not leave compiled binaries or other build artifacts in your codebase. Build in a temporary directory if you need to.

You may assume the input to `Compress` is **non-decreasing**: each value is greater than or equal to the one before it. Consecutive values may be **equal**, so a delta of zero is legal and must round-trip correctly. Do not assume the values are strictly increasing.

You may assume the input to `Decompress` is valid and was produced by a correct call to `Compress`.

## Evaluation Criteria

Your score is the average time `Decompress` takes, over several runs. Lower is better.

Compression ratio and compression time are both recorded alongside it, but neither is part of the score.
