package network

import (
	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

type SlowTwoLayerNet struct {
	Params map[string]num.Matrix
}

func NewSlowTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *SlowTwoLayerNet {
	params := map[string]num.Matrix{}
	W1 := num.NewRandnMatrix(inputSize, hiddenSize)
	W2 := num.NewRandnMatrix(hiddenSize, outputSize)
	params["W1"] = num.Mul(W1, weightInitStd)
	params["b1"] = num.Zeros(1, hiddenSize)
	params["W2"] = num.Mul(W2, weightInitStd)
	params["b2"] = num.Zeros(1, outputSize)
	return &SlowTwoLayerNet{Params: params}
}

func (net *SlowTwoLayerNet) Predict(x num.Matrix) num.Matrix {
	W1 := net.Params["W1"]
	b1 := net.Params["b1"]
	W2 := net.Params["W2"]
	b2 := net.Params["b2"]

	dota1 := num.Dot(x, W1)
	a1 := num.Add(dota1, b1.Flatten())
	z1 := num.Relu(a1)
	a2 := num.Add(num.Dot(z1, W2), b2.Flatten())
	y := num.Softmax(a2)

	return y
}

func (net *SlowTwoLayerNet) Loss(x, t num.Matrix) float64 {
	y := net.Predict(x)
	cee := num.CrossEntropyError(y, t)
	return cee
}

func (net *SlowTwoLayerNet) Accuracy(x, t num.Matrix) float64 {
	y := net.Predict(x)
	yMax := num.ArgMax(y, 1)
	tMax := num.ArgMax(t, 1)
	sum := 0.0
	r, _ := x.Shape()
	for i, v := range yMax {
		if v == tMax[i] {
			sum += 1.0
		}
	}
	accuracy := sum / float64(r)
	return accuracy
}

func (net *SlowTwoLayerNet) NumericalGradient(x, t num.Matrix) map[string]num.Matrix {
	lossW := func(wvec vec.Vector) float64 {
		return net.Loss(x, t)
	}
	grads := map[string]num.Matrix{}
	grads["W1"] = num.NumericalGradient(lossW, net.Params["W1"])
	grads["b1"] = num.NumericalGradient(lossW, net.Params["b1"])
	grads["W2"] = num.NumericalGradient(lossW, net.Params["W2"])
	grads["b2"] = num.NumericalGradient(lossW, net.Params["b2"])
	return grads
}
