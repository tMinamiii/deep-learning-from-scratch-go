package main

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/mnist"
	"github.com/naronA/zero_deeplearning/network"
)

func runBasicNetwork() {
	network := network.NewBasicNetwork()
	x, _ := mat.NewMat64(1, 2, array.Array{
		1.0, 0.5,
	})
	y := network.Forward(x)
	fmt.Println(y)
	train, _ := mnist.LoadMnist()
	for i := 0; i < 30; i++ {
		fmt.Println(train.Label[i])
	}
}
func runSimplwNetwork() {
	w, err := mat.NewMat64(2, 3, array.Array{
		0.47355232, 0.9977393, 0.84668094,
		0.85557411, 0.03563661, 0.69422093,
	})
	if err != nil {
		panic(err)
	}
	x, err := mat.NewMat64(1, 2, array.Array{0.6, 0.9})
	if err != nil {
		panic(err)
	}
	sn := network.NewSimpleNet(w)
	p := sn.Predict(x)
	fmt.Println(p)
	t := array.Array{0, 0, 1}
	loss := sn.Loss(x, t)
	fmt.Println(loss)
}
func main() {
	runSimplwNetwork()
}
