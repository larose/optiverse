package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const numRuns = 3

func loadTestData(filename string) ([]uint32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var result []uint32
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue // Skip empty lines
		}

		val, err := strconv.ParseUint(line, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid integer on line: %s", line)
		}

		result = append(result, uint32(val))
	}

	if err := scanner.Err(); err != nil {
		return nil, err
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

	// Run multiple times and collect metrics
	var decompressionTimes []int64
	var compressionTimes []int64
	var compressionRatios []float64

	for run := 0; run < numRuns; run++ {
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

		// Collect metrics
		decompressionTimes = append(decompressionTimes, decompressionTime.Milliseconds())
		compressionTimes = append(compressionTimes, compressionTime.Milliseconds())
		compressionRatios = append(compressionRatios, compressionRatio)
	}

	// Calculate averages
	var totalDecompressionTime int64
	var totalCompressionTime int64
	var totalCompressionRatio float64

	for i := 0; i < numRuns; i++ {
		totalDecompressionTime += decompressionTimes[i]
		totalCompressionTime += compressionTimes[i]
		totalCompressionRatio += compressionRatios[i]
	}

	avgDecompressionTime := float64(totalDecompressionTime) / float64(numRuns)
	avgCompressionTime := float64(totalCompressionTime) / float64(numRuns)
	avgCompressionRatio := totalCompressionRatio / float64(numRuns)

	// Output averaged metrics with >>> prefix
	fmt.Printf(">>> decompression_time: %.0f\n", avgDecompressionTime)
	fmt.Printf(">>> compression_ratio: %.3f\n", avgCompressionRatio)
	fmt.Printf(">>> compression_time: %.0f\n", avgCompressionTime)
}
