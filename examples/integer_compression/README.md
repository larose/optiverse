# Integer Compression

## Problem Summary

The goal is to implement a compression algorithm for sorted 32-bit unsigned integers in Go, optimizing primarily decompression speed.

See [problem.md](problem.md) for the complete problem description and requirements.

## Layout

- [optimize.py](optimize.py) — starts an optimization run. `make run.integer_compression` runs it.
- [initial/](initial/) — the seed codebase: a `compressor.go` that serialises the values unchanged. The first solution is a copy of this, and every later one descends from it. The evolvable unit is the whole package, not one file.
- [harness/](harness/) — everything the example owns for measuring. Never copied into a codebase and never visible to an agent.
  - `evaluate.py` — the evaluator command, and the only thing here anyone else calls. `score` builds the candidate and takes three timings; `validate` builds it and round-trips the synthetic cases.
  - `score/main.go` — compresses the dataset, times `Decompress`, writes what it measured.
  - `validate/main.go` — checks `Decompress(Compress(data)) == data` across eighteen synthetic cases: block boundaries at 127/128/129, values at `MaxUint32`, and repeated values, which are what the real dataset is almost entirely made of. It needs no dataset and finishes in milliseconds, which is what makes it usable inside the agent's loop.
  - `dataset.py` — fetches `ts.txt` and converts it once into `ts.bin`.
  - `ts.txt`, `ts.bin` — the benchmark data. Gitignored, and they live here and nowhere else.

The agent has no way to run any of it. It writes Go and calls `validate`, which
answers valid or invalid with diagnostics and nothing more.

Scoring needs the benchmark dataset (~1.6 GB downloaded, ~577 MB converted, both gitignored):

```bash
make run.integer_compression   # fetches and converts on first run
```

## How a candidate is measured

`evaluate.py` builds a workspace, compiles it, and deletes it afterwards:

```
workspace/
  go.mod  main.go     <- from harness/, either score/ or validate/
  candidate/          <- the codebase, verbatim
  bench               <- the built binary
```

The candidate is its own Go package, imported by `main.go` as
`harness/candidate`. Being a separate package is what makes it safe to let a
codebase be a directory of arbitrary files: nothing it contains can stand in for
anything the harness owns, whatever it names its files, so there is no reserved
name list and nothing has to be overwritten.

Scoring compiles once and then runs the binary three times, each in a directory
of its own with the environment scrubbed. Separate processes, so a result cached
on one run is not there for the next, and the timing is the average of the three.

The binary writes its timings to a file rather than printing them, so a candidate
debugging on stdout cannot corrupt its own score. On the first run it also writes
out the compressed bytes, which `evaluate.py` sizes itself, so the compression
ratio is a measurement rather than a number the harness was asked to report —
once, because half a gigabyte through `TMPDIR` on every run costs more than it
measures. Ratio is recorded, not scored: the score is decompression time alone.

Two things are *not* closed, both for the same reason: `Compress` and `Decompress`
are compiled into the binary that measures them, and they run in one process
because the second is called on what the first returned.

So a candidate could report a time it did not achieve — the clock is code it
shares a binary with. And it could keep the input in a package-level variable
during `Compress` and hand it back from `Decompress`, which forges nothing and
simply is fast. `problem.md` requires `Decompress` to reconstruct the values from
the bytes it is given, which rules the second out by rule rather than by
structure.

Closing either means measuring from outside, or splitting the two calls across
two processes. For a benchmark this short both cost more than the measurement is
worth, so what is left is that a deliberate cheat succeeds and is plainly visible
in the winning solution's source. The TSP example takes the opposite tradeoff,
because a tour length is something its evaluator can simply recompute.

## Solution Found by Optiverse

After approximately 1000 iterations, using Qwen3-235B-A22B as the LLM, Optiverse generated a highly efficient block-based delta encoding combined with binary packing.


## Benchmark Results

Performance was benchmarked against well-known C implementations using this [benchmark suite](https://github.com/vteromero/integer-compression-benchmarks)

These numbers were measured under the previous harness, on the CPU named below,
parsing `ts.txt` directly and running its three repetitions inside one process.
The current harness reads `ts.bin` and gives each repetition its own process, so
each run pays for faulting in its own output rather than amortising that across
the three. Figures produced now are therefore not comparable with the table
below — only with each other, which is all the search needs.

For reference, on the machine this was reworked on, the current harness scores
the seed at ~653 ms (ratio 1.0) and the evolved solution below at ~459 ms
(ratio 230.0):

```bash
python examples/integer_compression/harness/evaluate.py score <codebase>
```


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
package candidate

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
