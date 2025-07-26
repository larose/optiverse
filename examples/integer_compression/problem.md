Design and implement a compression algorithm for sorted 32-bit unsigned integers in Go by completing the two function stubs provided below.

Your primary goal is to achieve the lowest possible decompression time while also maintaining a competitive compression ratio and reasonable compression time.

To be competitive, your solution must surpass traditional methods like Variable Byte (VByte), PForDelta, and VTEnc. Drawing inspiration from these or related techniques is allowed, but mere reimplementation is insufficient. True innovation is required.

You are encouraged to explore novel approaches based on well-known general patterns, including but not limited to:
    - Delta Encoding and Delta-of-Delta
    - Bit-Packing and Frame-of-Reference
    - Headerless or Self-Describing Formats
    - Table-Driven and SIMD-Accelerated Decoding
    - Block-based or chunked compression with skippable blocks

## Requirements

Implement the following two functions with these exact signatures:

```go
func Compress(data []uint32) []byte
func Decompress(compressed []byte) []uint32
```

You may define additional package-level helper functions (no nested functions).

Use only the Go Standard Library. Third-party packages are not allowed.

The use of built-in compression libraries (compress/gzip, flate, zlib, lzw, etc.) is strictly prohibited.

The output of `Decompress(Compress(data))` must exactly match the original `data`.

You may assume that:
    - The input to `Compress` (`data`) is valid and strictly increasing.
    - The input to `Decompress` (`compressed`) is valid and was produced by a correct call to `Compress`.
