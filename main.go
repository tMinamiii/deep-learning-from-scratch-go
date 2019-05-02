package main

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/network"
)

func calcNetwork(wArray array.Array) float64 {
	w, err := mat.NewMat64(2, 3, wArray)
	if err != nil {
		panic(err)
	}
	x, err := mat.NewMat64(1, 2, array.Array{0.6, 0.9})
	if err != nil {
		panic(err)
	}

	sn := network.NewSimpleNet(w)
	t, _ := mat.NewMat64(1, 3, array.Array{0, 0, 1})
	return sn.Loss(x, t)
}

func main() {
	net := network.NewTwoLayerNet(784, 100, 10, 0.01)
	fmt.Println(net.Params["W1"].Shape())
	fmt.Println(net.Params["b1"].Shape())
	fmt.Println(net.Params["W2"].Shape())
	fmt.Println(net.Params["b2"].Shape())
	x, _ := mat.NewRandnMat64(100, 784) // ダミーの入力ラベル
	t, _ := mat.NewRandnMat64(100, 10) // ダミーの正解ラベル
	fmt.Println(x.Shape())
	p := net.Predict(x)
	fmt.Println(p)
	grads := net.NumericalGradient(x, t)
	fmt.Println(grads["W1"].Shape())
	fmt.Println(grads["b1"].Shape())
	fmt.Println(grads["W2"].Shape())
	fmt.Println(grads["b2"].Shape())
}
