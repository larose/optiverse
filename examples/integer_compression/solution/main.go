package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"time"
)

func loadTestData(filename string) ([]uint32, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	if len(data)%4 != 0 {
		return nil, fmt.Errorf("invalid data file: size %d is not divisible by 4", len(data))
	}

	var result []uint32
	for i := 0; i < len(data); i += 4 {
		val := binary.LittleEndian.Uint32(data[i : i+4])
		result = append(result, val)
	}

	return result, nil
}

func calculateOriginalSize(data []uint32) int {
	return len(data) * 4 // 4 bytes per uint32
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <test_data_file>\n", os.Args[0])
		os.Exit(1)
	}

	// Load test data
	data, err := loadTestData(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		os.Exit(1)
	}

	originalSize := calculateOriginalSize(data)

	// Measure compression
	start := time.Now()
	compressed := Compress(data)
	compressionTime := time.Since(start)
	compressedSize := len(compressed)

	// Measure decompression
	start = time.Now()
	decompressed := Decompress(compressed)
	decompressionTime := time.Since(start)

	// Verify correctness
	if len(decompressed) != len(data) {
		fmt.Fprintf(os.Stderr, "Length mismatch: got %d, expected %d\n",
			len(decompressed), len(data))
		os.Exit(1)
	}
	for j := range data {
		if decompressed[j] != data[j] {
			fmt.Fprintf(os.Stderr, "Data mismatch at index %d: got %d, expected %d\n",
				j, decompressed[j], data[j])
			os.Exit(1)
		}
	}

	compressionRatio := float64(originalSize) / float64(compressedSize)

	// Output metrics with >>> prefix
	fmt.Printf(">>> decompression_time: %d\n", decompressionTime.Milliseconds())
	fmt.Printf(">>> compression_ratio: %.3f\n", compressionRatio)
	fmt.Printf(">>> compression_time: %d\n", compressionTime.Milliseconds())
}
