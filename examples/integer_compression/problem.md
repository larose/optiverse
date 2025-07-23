Implement a compression algorithm for 32-bit unsigned integers in Go by completing the functions below. Your goal is to achieve the best decompression speed while maintaining good compression ratio.

## Requirements

Define `Compress` and `Decompress` functions with the following signatures:

```go
func Compress(data []uint32) ([]byte, error)
func Decompress(compressed []byte) ([]uint32, error)
```

Define any helper functions at the package level (do not nest functions).

Use only the Go Standard Library; external packages are not allowed.

Built-in compression packages (compress/gzip, compress/flate, compress/zlib, compress/lzw) are prohibited.

The decompressed data must exactly match the original input data.

Handle errors appropriately and return them.

## Evaluation Criteria

Your score is the average decompression time in nanoseconds. Lower is better.

Secondary metrics include compression ratio and compression time.

## Test Data

Your solution will be tested on arrays of 32-bit unsigned integers with various patterns:
- Sequential numbers (1, 2, 3, ...)
- Random order numbers
- Numbers with duplicates
- Large ranges of values
