# Integer Compression

## Problem Summary

The goal is to implement a compression algorithm for sorted 32-bit unsigned integers in Go, optimizing primarily decompression speed.

See [problem.md](problem.md) for the complete problem description and requirements.

## Solution Found by Optiverse

After approximately 1000 iterations, using Qwen3-235B-A22B as the LLM, Optiverse generated a highly efficient block-based delta encoding combined with binary packing.


## Benchmark Results

Performance was benchmarked against well-known C implementations using this [benchmark suite](https://github.com/vteromero/integer-compression-benchmarks)


| Algorithm          | Decoding Speed (GB/s) | Compression Ratio |
|-----------------------------|--------------|-------------------|
| VTEnc                        | 8.1          | 26,622 |
| Delta+VarIntGB              | 7.9          | 3.2 |
| **Delta+BinaryPacking (Optiverse)**       | **5.7**      | **230** |
| Delta+BinaryPacking (C)         | 5.2          | 127 |
| Delta+FastPFor128            | 4.4          | 250 |
| Delta+FastPFor256            | 4.2          | 489 |
| Delta+VariableByte          | 4.4          | 4 |

(Higher is better for both metrics)

CPU: Intel Core i5-6500T @ 2.50GHz

## Full Solution Code

```go
package main

import (
	"encoding/binary"
	"math/bits"
)

const blockSize = 128

func Compress(data []uint32) []byte {
	if len(data) == 0 {
		return nil
	}

	compressed := make([]byte, 8)
	binary.LittleEndian.PutUint32(compressed[0:4], data[0])
	if len(data) == 1 {
		return compressed[:4]
	}

	deltas := make([]uint32, len(data)-1)
	for i := range deltas {
		deltas[i] = data[i+1] - data[i]
	}
	binary.LittleEndian.PutUint32(compressed[4:8], uint32(len(deltas)))

	pos := 8
	for i := 0; i < len(deltas); i += blockSize {
		end := i + blockSize
		if end > len(deltas) {
			end = len(deltas)
		}
		block := deltas[i:end]

		maxDelta := uint32(0)
		for _, d := range block {
			if d > maxDelta {
				maxDelta = d
			}
		}

		bitsPerDelta := 0
		if maxDelta > 0 {
			bitsPerDelta = bits.Len32(maxDelta)
		}
		bytesPerDelta := 0
		if bitsPerDelta > 0 {
			bytesPerDelta = (bitsPerDelta + 7) / 8
		}

		if pos+1 > cap(compressed) {
			newCap := cap(compressed) * 2
			if newCap == 0 {
				newCap = 64
			}
			newBuf := make([]byte, len(compressed), newCap)
			copy(newBuf, compressed)
			compressed = newBuf
		}
		compressed = compressed[:pos+1]
		compressed[pos] = byte(bytesPerDelta)
		pos++

		if bytesPerDelta == 0 {
			continue
		}

		packedBytes := len(block) * bytesPerDelta
		if pos+packedBytes > cap(compressed) {
			newCap := cap(compressed)
			for newCap < pos+packedBytes {
				newCap *= 2
			}
			newBuf := make([]byte, len(compressed), newCap)
			copy(newBuf, compressed)
			compressed = newBuf
		}
		compressed = compressed[:pos+packedBytes]

		idx := 0
		switch bytesPerDelta {
		case 1:
			for _, d := range block {
				compressed[pos+idx] = byte(d)
				idx++
			}
		case 2:
			for _, d := range block {
				binary.LittleEndian.PutUint16(compressed[pos+idx:pos+idx+2], uint16(d))
				idx += 2
			}
		case 3:
			for _, d := range block {
				compressed[pos+idx] = byte(d)
				compressed[pos+idx+1] = byte(d >> 8)
				compressed[pos+idx+2] = byte(d >> 16)
				idx += 3
			}
		case 4:
			for _, d := range block {
				binary.LittleEndian.PutUint32(compressed[pos+idx:pos+idx+4], d)
				idx += 4
			}
		}

		pos += packedBytes
	}

	return compressed[:pos]
}

func Decompress(compressed []byte) []uint32 {
	if len(compressed) < 8 {
		if len(compressed) < 4 {
			return []uint32{}
		}
		first := binary.LittleEndian.Uint32(compressed[0:4])
		return []uint32{first}
	}

	first := binary.LittleEndian.Uint32(compressed[0:4])
	numDeltas := binary.LittleEndian.Uint32(compressed[4:8])
	totalLen := int(numDeltas) + 1
	result := make([]uint32, totalLen)
	if totalLen == 1 {
		return result[:1]
	}

	result[0] = first
	pos := 8
	remaining := int(numDeltas)
	prev := first
	currentIdx := 1

	for remaining > 0 {
		bytesPerDelta := int(compressed[pos])
		pos++

		blockSizeCurrent := blockSize
		if remaining < blockSize {
			blockSizeCurrent = remaining
		}
		packedBytes := blockSizeCurrent * bytesPerDelta

		switch bytesPerDelta {
		case 0:
			end := currentIdx + blockSizeCurrent
			value := prev
			for i := currentIdx; i < end; i++ {
				result[i] = value
			}
			currentIdx = end

		case 1:
			p := pos
			endp := pos + blockSizeCurrent
			for p < endp {
				d := uint32(compressed[p])
				prev += d
				result[currentIdx] = prev
				currentIdx++
				p++
			}

		case 2:
			p := pos
			endp := pos + 2*blockSizeCurrent
			for p < endp {
				d := uint32(binary.LittleEndian.Uint16(compressed[p:]))
				prev += d
				result[currentIdx] = prev
				currentIdx++
				p += 2
			}

		case 3:
			p := pos
			endp := pos + 3*blockSizeCurrent
			for p < endp {
				d := uint32(compressed[p]) | (uint32(compressed[p+1]) << 8) | (uint32(compressed[p+2]) << 16)
				prev += d
				result[currentIdx] = prev
				currentIdx++
				p += 3
			}

		case 4:
			p := pos
			endp := pos + 4*blockSizeCurrent
			for p < endp {
				d := binary.LittleEndian.Uint32(compressed[p:])
				prev += d
				result[currentIdx] = prev
				currentIdx++
				p += 4
			}
		}

		remaining -= blockSizeCurrent
		pos += packedBytes
	}

	return result
}
```
