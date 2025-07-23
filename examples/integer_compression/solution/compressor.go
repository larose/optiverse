package main

import (
	"encoding/binary"
	"errors"
)

func Compress(data []uint32) ([]byte, error) {
	// Simple baseline: no compression, just serialize as-is
	result := make([]byte, len(data)*4)

	for i, val := range data {
		binary.LittleEndian.PutUint32(result[i*4:(i+1)*4], val)
	}

	return result, nil
}

func Decompress(compressed []byte) ([]uint32, error) {
	// Simple baseline: just deserialize the data
	if len(compressed)%4 != 0 {
		return nil, errors.New("invalid compressed data: length not divisible by 4")
	}

	result := make([]uint32, len(compressed)/4)

	for i := 0; i < len(result); i++ {
		result[i] = binary.LittleEndian.Uint32(compressed[i*4:(i+1)*4])
	}

	return result, nil
}
