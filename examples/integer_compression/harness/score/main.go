// Scoring harness: compress the dataset, time decompression, report.
//
// One measurement per process. The evaluator runs this binary several times and
// averages, so nothing a candidate parks in package state on one run is there
// for the next.
//
// The candidate is a separate package, imported here. It cannot redefine
// anything in this file, and it is never handed the dataset path -- that arrives
// on this process's command line and is read before any candidate code runs.
//
// Measurements go to a file rather than stdout. A candidate is free to print
// whatever it likes while debugging without corrupting its own score, and the
// compressed bytes are written out so the evaluator can size them itself instead
// of trusting a ratio this process reports.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"harness/candidate"
)

// Read in chunks rather than slurping the file: the dataset is ~577 MB and the
// decoded slice is another 577 MB, so holding both at once doubles the peak.
const readBufferSize = 1 << 20

type measurements struct {
	CompressionTimeMS   float64 `json:"compression_time_ms"`
	DecompressionTimeMS float64 `json:"decompression_time_ms"`
	ValueCount          int     `json:"value_count"`
}

func readValues(path string) ([]uint32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return nil, err
	}

	size := info.Size()
	if size%4 != 0 {
		return nil, fmt.Errorf("%s holds %d bytes, not a whole number of uint32s", path, size)
	}

	values := make([]uint32, size/4)
	buffer := make([]byte, readBufferSize)
	index := 0

	for {
		n, err := io.ReadFull(file, buffer)

		for offset := 0; offset+4 <= n; offset += 4 {
			values[index] = binary.LittleEndian.Uint32(buffer[offset:])
			index++
		}

		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}

	if index != len(values) {
		return nil, fmt.Errorf("%s: read %d of %d values", path, index, len(values))
	}

	return values, nil
}

// verify checks the round trip. Reported before any timing is trusted, because
// a fast wrong answer is not an answer.
func verify(original, decompressed []uint32) error {
	if len(decompressed) != len(original) {
		return fmt.Errorf(
			"length mismatch: got %d values, want %d", len(decompressed), len(original),
		)
	}

	for i := range original {
		if decompressed[i] != original[i] {
			return fmt.Errorf(
				"value mismatch at index %d: got %d, want %d",
				i, decompressed[i], original[i],
			)
		}
	}

	return nil
}

func write(path string, compressed []byte, measured measurements) error {
	if err := os.WriteFile(path+".bin", compressed, 0o644); err != nil {
		return err
	}

	encoded, err := json.Marshal(measured)
	if err != nil {
		return err
	}

	return os.WriteFile(path+".json", encoded, 0o644)
}

func run() error {
	instancePath := flag.String("instance", "", "dataset, raw little-endian uint32")
	outputPrefix := flag.String("output", "", "prefix for <prefix>.bin and <prefix>.json")
	flag.Parse()

	if *instancePath == "" || *outputPrefix == "" {
		return fmt.Errorf("both -instance and -output are required")
	}

	data, err := readValues(*instancePath)
	if err != nil {
		return err
	}

	start := time.Now()
	compressed := candidate.Compress(data)
	compressionTime := time.Since(start)

	start = time.Now()
	decompressed := candidate.Decompress(compressed)
	decompressionTime := time.Since(start)

	if err := verify(data, decompressed); err != nil {
		return err
	}

	return write(*outputPrefix, compressed, measurements{
		CompressionTimeMS:   float64(compressionTime.Nanoseconds()) / 1e6,
		DecompressionTimeMS: float64(decompressionTime.Nanoseconds()) / 1e6,
		ValueCount:          len(data),
	})
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
