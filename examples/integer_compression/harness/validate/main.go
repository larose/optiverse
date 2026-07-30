// Validation harness: checks that Decompress(Compress(data)) == data.
//
// Deliberately says nothing about speed or compression ratio. It runs on small
// synthetic inputs so it finishes in milliseconds and needs none of the
// benchmark data, which is what makes it usable inside an agent's loop.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
)

// strictlyIncreasing builds n strictly increasing uint32s with the given
// average gap. Compress may assume its input is strictly increasing.
func strictlyIncreasing(n int, averageGap uint32, seed int64) []uint32 {
	if n == 0 {
		return nil
	}

	random := rand.New(rand.NewSource(seed))
	data := make([]uint32, n)

	var current uint64
	for i := range data {
		gap := uint64(1)
		if averageGap > 1 {
			gap = uint64(1 + random.Intn(int(averageGap)*2))
		}

		current += gap
		if current > math.MaxUint32 {
			// Saturate rather than wrap: the input must stay increasing.
			current = math.MaxUint32 - uint64(n-i)
		}

		data[i] = uint32(current)
	}

	return data
}

func uniformGap(n int, gap uint32) []uint32 {
	data := make([]uint32, n)
	for i := range data {
		data[i] = uint32(i+1) * gap
	}
	return data
}

type testCase struct {
	name string
	data []uint32
}

func cases() []testCase {
	return []testCase{
		{"empty", nil},
		{"single", []uint32{42}},
		{"two", []uint32{1, 2}},
		// Block-boundary sizes: implementations commonly chunk at 128.
		{"block-minus-one", strictlyIncreasing(127, 4, 1)},
		{"block-exact", strictlyIncreasing(128, 4, 2)},
		{"block-plus-one", strictlyIncreasing(129, 4, 3)},
		{"two-blocks-plus-remainder", strictlyIncreasing(300, 7, 4)},
		{"tiny-gaps", strictlyIncreasing(1000, 1, 5)},
		{"wide-gaps", strictlyIncreasing(1000, 100000, 6)},
		{"uniform-gap", uniformGap(512, 3)},
		{"first-value-zero", append([]uint32{0}, strictlyIncreasing(200, 5, 7)...)},
		{"near-max", []uint32{math.MaxUint32 - 2, math.MaxUint32 - 1, math.MaxUint32}},
	}
}

func check(name string, data []uint32) error {
	compressed := Compress(data)
	decompressed := Decompress(compressed)

	if len(decompressed) != len(data) {
		return fmt.Errorf(
			"%s: length mismatch: got %d values, want %d",
			name, len(decompressed), len(data),
		)
	}

	for i := range data {
		if decompressed[i] != data[i] {
			return fmt.Errorf(
				"%s: value mismatch at index %d: got %d, want %d",
				name, i, decompressed[i], data[i],
			)
		}
	}

	return nil
}

func main() {
	failures := 0

	for _, testCase := range cases() {
		if err := check(testCase.name, testCase.data); err != nil {
			fmt.Fprintf(os.Stderr, "FAIL %v\n", err)
			failures++
			continue
		}

		fmt.Fprintf(os.Stderr, "ok   %s (%d values)\n", testCase.name, len(testCase.data))
	}

	if failures > 0 {
		fmt.Fprintf(os.Stderr, "\n%d round-trip failure(s)\n", failures)
		os.Exit(1)
	}
}
