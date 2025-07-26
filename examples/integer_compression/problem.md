Design and implement a compression algorithm for sorted 32-bit unsigned integers in Go by completing the two function stubs provided below. Your primary objective is to achieve maximum decompression speed (lower is better), while also maintaining a competitive compression ratio.

To be competitive, you must go beyond standard techniques such as Variable Byte (VByte), PForDelta, or VTEnc. You may draw inspiration from these and related schemes, but true innovation is required. Simply re-implementing known algorithms will not suffice.

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

You may assume that both:
  - The input to `Compress` (`data`) is valid and strictly increasing.
  - The input to `Decompress` (`compressed`) is valid and was produced by `Compress`.
