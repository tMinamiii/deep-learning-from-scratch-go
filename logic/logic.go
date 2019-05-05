package logic

import (
	"log"

	"github.com/naronA/zero_deeplearning/vec"
)

// AND is 論理和
func AND(x1 float64, x2 float64) int {
	x := vec.Vector{x1, x2}
	w := vec.Vector{0.5, 0.5}
	const b = -0.7
	mul := vec.Mul(x, w)
	if mul == nil {
		return 0
	}
	tmp := vec.Sum(mul) + b
	if tmp <= 0 {
		return 0
	}
	return 1

}

// NAND is 論理和
func NAND(x1 float64, x2 float64) int {
	x := vec.Vector{x1, x2}
	w := vec.Vector{-0.5, -0.5}
	const b = 0.7
	mul := vec.Mul(x, w)
	if mul == nil {
		return 0
	}
	tmp := vec.Sum(mul) + b
	if tmp <= 0 {
		return 0
	}
	return 1
}

// OR is 論理和
func OR(x1 float64, x2 float64) int {
	x := vec.Vector{x1, x2}
	w := vec.Vector{0.5, 0.5}
	const b = -0.2
	mul := vec.Mul(x, w)
	if mul == nil {
		return 0
	}
	tmp := vec.Sum(mul) + b
	if tmp <= 0 {
		return 0
	}
	return 1
}

// OR is 論理和
func XOR(x1 float64, x2 float64) int {
	s1 := NAND(x1, x2)
	s2 := OR(x1, x2)
	y := AND(float64(s1), float64(s2))
	log.Println(y)
	return y
}
