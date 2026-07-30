// Validation harness: checks that Decompress(Compress(data)) == data.
//
// Deliberately says nothing about speed or compression ratio. It runs on small
// synthetic inputs so it finishes in milliseconds and needs none of the
// benchmark data, which is what makes it usable inside an agent's loop.
//
// The candidate is a separate package, imported here, so nothing it contains can
// stand in for anything in this file.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"harness/candidate"
)

// nonDecreasing builds n non-decreasing uint32s with the given average gap.
// Gaps of zero are legal and, in the scoring dataset, are the common case.
func nonDecreasing(n int, averageGap uint32, seed int64) []uint32 {
	if n == 0 {
		return nil
	}

	random := rand.New(rand.NewSource(seed))
	data := make([]uint32, n)

	var current uint64
	for i := range data {
		gap := uint64(0)
		if averageGap > 0 {
			gap = uint64(random.Intn(int(averageGap)*2 + 1))
		}

		current += gap
		if current > math.MaxUint32 {
			// Saturate rather than wrap: the input must stay non-decreasing.
			current = math.MaxUint32
		}

		data[i] = uint32(current)
	}

	return data
}

// increasing builds n strictly increasing uint32s: the case with no repeats at
// all, which must keep working too.
func increasing(n int, averageGap uint32, seed int64) []uint32 {
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

// runs builds long stretches of one repeated value, then a step. This is the
// shape of the scoring dataset, where 99.99% of consecutive pairs are equal.
func runs(runCount int, runLength int, step uint32) []uint32 {
	data := make([]uint32, 0, runCount*runLength)

	value := uint32(0)
	for run := 0; run < runCount; run++ {
		for i := 0; i < runLength; i++ {
			data = append(data, value)
		}
		value += step
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
		{"block-minus-one", increasing(127, 4, 1)},
		{"block-exact", increasing(128, 4, 2)},
		{"block-plus-one", increasing(129, 4, 3)},
		{"two-blocks-plus-remainder", increasing(300, 7, 4)},
		{"tiny-gaps", increasing(1000, 1, 5)},
		{"wide-gaps", increasing(1000, 100000, 6)},
		{"uniform-gap", uniformGap(512, 3)},
		{"first-value-zero", append([]uint32{0}, increasing(200, 5, 7)...)},
		{"near-max", []uint32{math.MaxUint32 - 2, math.MaxUint32 - 1, math.MaxUint32}},
		// Repeated values. A solver that assumes strictly increasing input fails
		// from here down, and would fail on almost every pair of the real data.
		{"all-equal", []uint32{7, 7, 7, 7, 7, 7, 7, 7}},
		{"equal-across-a-block", runs(2, 130, 11)},
		{"long-runs", runs(40, 25, 1)},
		{"mostly-equal", nonDecreasing(1000, 0, 8)},
		{"occasional-step", nonDecreasing(1000, 1, 9)},
		{"repeats-at-max", []uint32{math.MaxUint32, math.MaxUint32, math.MaxUint32}},
	}
}

func check(name string, data []uint32) error {
	compressed := candidate.Compress(data)
	decompressed := candidate.Decompress(compressed)

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
