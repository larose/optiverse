package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math/rand"
	"os"
	"time"
)

// Expected hashes for deterministic test data files
var expectedHashes = map[string]string{
	"test_data_small.bin":  "",  // Will be computed on first run
	"test_data_medium.bin": "",  // Will be computed on first run
	"test_data_large.bin":  "",  // Will be computed on first run
}

func generateTestData(filename string, seed int64, size int) error {
	rng := rand.New(rand.NewSource(seed))
	data := make([]uint32, size)

	for i := 0; i < size; i++ {
		switch filename {
		case "test_data_small.bin":
			// Sequential pattern for small data
			data[i] = uint32(i + 1)
		case "test_data_medium.bin":
			// Random values for medium data
			data[i] = rng.Uint32() % 1000000
		case "test_data_large.bin":
			// Mixed patterns for large data
			if i%3 == 0 {
				data[i] = uint32(i/3 + 1) // Some duplicates
			} else if i%3 == 1 {
				data[i] = rng.Uint32() % 10000000 // Random values
			} else {
				data[i] = uint32(i) // Sequential
			}
		}
	}

	return writeUint32Array(filename, data)
}

func writeUint32Array(filename string, data []uint32) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, val := range data {
		bytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(bytes, val)
		if _, err := file.Write(bytes); err != nil {
			return err
		}
	}

	return nil
}

func calculateFileHash(filename string) (string, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return "", err
	}

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

func ensureTestDataExists(filename string) error {
	// Define seed and size for each file
	var seed int64
	var size int

	switch filename {
	case "test_data_small.bin":
		seed = 12345
		size = 1000
	case "test_data_medium.bin":
		seed = 67890
		size = 10000
	case "test_data_large.bin":
		seed = 54321
		size = 100000
	default:
		return fmt.Errorf("unknown test data file: %s", filename)
	}

	// Check if file exists and has correct hash
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		fmt.Printf("Generating %s...\n", filename)
		if err := generateTestData(filename, seed, size); err != nil {
			return fmt.Errorf("failed to generate %s: %v", filename, err)
		}
	}

	// Verify hash if we have an expected hash
	currentHash, err := calculateFileHash(filename)
	if err != nil {
		return fmt.Errorf("failed to calculate hash for %s: %v", filename, err)
	}

	expectedHash := expectedHashes[filename]
	if expectedHash == "" {
		// First time - store the hash
		expectedHashes[filename] = currentHash
		fmt.Printf("Stored hash for %s: %s\n", filename, currentHash)
	} else if currentHash != expectedHash {
		// Hash mismatch - regenerate
		fmt.Printf("Hash mismatch for %s, regenerating...\n", filename)
		if err := generateTestData(filename, seed, size); err != nil {
			return fmt.Errorf("failed to regenerate %s: %v", filename, err)
		}
	}

	return nil
}

func loadTestData(filename string) ([]uint32, error) {
	// Ensure the test data file exists with correct hash
	if err := ensureTestDataExists(filename); err != nil {
		return nil, err
	}

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
	compressed, err := Compress(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Compression error: %v\n", err)
		os.Exit(1)
	}
	compressionTime := time.Since(start)
	compressedSize := len(compressed)

	// Measure decompression (multiple runs for accuracy)
	const numRuns = 10
	var totalDecompTime time.Duration

	for i := 0; i < numRuns; i++ {
		start = time.Now()
		decompressed, err := Decompress(compressed)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Decompression error: %v\n", err)
			os.Exit(1)
		}
		totalDecompTime += time.Since(start)

		// Verify correctness on first run
		if i == 0 {
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
		}
	}

	avgDecompTime := totalDecompTime / numRuns
	compressionRatio := float64(originalSize) / float64(compressedSize)

	// Output metrics with >>> prefix
	fmt.Printf(">>> decompression_time: %d\n", avgDecompTime.Nanoseconds())
	fmt.Printf(">>> compression_ratio: %.3f\n", compressionRatio)
	fmt.Printf(">>> compression_time: %d\n", compressionTime.Nanoseconds())
}
