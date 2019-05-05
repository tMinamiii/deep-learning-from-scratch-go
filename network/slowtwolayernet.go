package network

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type SlowTwoLayerNet struct {
	Params map[string]*mat.Matrix
}

func NewSlowTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *SlowTwoLayerNet {
	params := map[string]*mat.Matrix{}
	W1, err := mat.NewRandnMatrix(inputSize, hiddenSize)
	if err != nil {
		panic(err)
	}
	W2, err := mat.NewRandnMatrix(hiddenSize, outputSize)
	if err != nil {
		panic(err)
	}
	params["W1"] = mat.Mul(W1, weightInitStd)
	params["b1"] = mat.Zeros(1, hiddenSize)
	params["W2"] = mat.Mul(W2, weightInitStd)
	params["b2"] = mat.Zeros(1, outputSize)
	return &SlowTwoLayerNet{Params: params}
}

func (net *SlowTwoLayerNet) Predict(x *mat.Matrix) *mat.Matrix {
	W1 := net.Params["W1"]
	b1 := net.Params["b1"]
	W2 := net.Params["W2"]
	b2 := net.Params["b2"]

	dota1 := mat.Dot(x, W1)
	a1 := mat.Add(dota1, b1.Vector)
	z1 := mat.Relu(a1)
	a2 := mat.Add(mat.Dot(z1, W2), b2.Vector)
	y := mat.Softmax(a2)

	return y
}

func (net *SlowTwoLayerNet) Loss(x, t *mat.Matrix) float64 {
	y := net.Predict(x)
	cee := mat.CrossEntropyError(y, t)
	return cee
}

func (net *SlowTwoLayerNet) Accuracy(x, t *mat.Matrix) float64 {
	y := net.Predict(x)
	yMax := mat.ArgMax(y, 1)
	tMax := mat.ArgMax(t, 1)
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

func (net *SlowTwoLayerNet) NumericalGradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	lossW := func(wvec vec.Vector) float64 {
		return net.Loss(x, t)
	}
	grads := map[string]*mat.Matrix{}
	grads["W1"] = mat.NumericalGradient(lossW, net.Params["W1"])
	grads["b1"] = mat.NumericalGradient(lossW, net.Params["b1"])
	grads["W2"] = mat.NumericalGradient(lossW, net.Params["W2"])
	grads["b2"] = mat.NumericalGradient(lossW, net.Params["b2"])
	return grads
}
