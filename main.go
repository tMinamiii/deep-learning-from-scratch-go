package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	m1 := mat.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	m2 := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
	var m mat.Dense
	m.Mul(m1, m2)
	fmt.Println(mat.Formatted(&m))
}
