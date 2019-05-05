package main

import (
	"fmt"
)

func main() {
	b := []byte{0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x10, 0x11, 0x12, 0x99}
	fmt.Println(b)
	f := make([]float64, len(b))
	for i, v := range b {
		f[i] = float64(v)
	}
	fmt.Println(f)
	// sss := []string{"affine1", "rerlu1", "affine2"}
}
