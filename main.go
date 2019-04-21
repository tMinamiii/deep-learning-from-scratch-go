package main

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
)

func initNetwork() map[string]*mat.Mat64 {
	network := map[string]*mat.Mat64{}

	network["W1"], _ = mat.NewMat64(2, 3, []float64{
		0.1, 0.3, 0.5,
		0.2, 0.4, 0.6,
	})
	network["b1"], _ = mat.NewMat64(1, 3, []float64{
		0.1, 0.2, 0.3,
	})
	network["W2"], _ = mat.NewMat64(3, 2, []float64{
		0.1, 0.4,
		0.2, 0.5,
		0.3, 0.6,
	})
	network["b2"], _ = mat.NewMat64(1, 2, []float64{
		0.1, 0.2,
	})
	network["W3"], _ = mat.NewMat64(2, 2, []float64{
		0.1, 0.3,
		0.2, 0.4,
	})
	network["b3"], _ = mat.NewMat64(1, 2, []float64{
		0.1, 0.2,
	})

	return network

}

func forward(network map[string]*mat.Mat64, x *mat.Mat64) []float64 {
	W1 := network["W1"]
	W2 := network["W2"]
	W3 := network["W3"]
	b1 := network["b1"]
	b2 := network["b2"]
	b3 := network["b3"]

	mul1 := mat.Mul(x, W1)
	a1 := mat.Add(mul1, b1)
	z1 := mat.Sigmoid(a1)

	mul2 := mat.Mul(z1, W2)
	a2 := mat.Add(mul2, b2)
	z2 := mat.Sigmoid(a2)

	mul3 := mat.Mul(z2, W3)
	a3 := mat.Add(mul3, b3)
	y := array.IdentityFunction(a3.Array)
	return y
}

func main() {
	network := initNetwork()
	x, _ := mat.NewMat64(1, 2, []float64{
		1.0, 0.5,
	})
	y := forward(network, x)
	fmt.Println(y)
}
