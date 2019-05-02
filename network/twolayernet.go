package network

import (
	"math/rand"
	"time"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
)

type TwoLayerNet struct {
	Params map[string]*mat.Mat64
}

func NewTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) {
	rand.Seed(time.Now().UnixNano())
	params := map[string]*mat.Mat64{}
	W1, err := mat.NewRandnMat64(inputSize, outputSize)
	if err != nil {
		panic(err)
	}
	W2, err := mat.NewRandnMat64(inputSize, outputSize)
	if err != nil {
		panic(err)
	}
	params["W1"] = W1.MulAll(weightInitStd)
	params["b1"] = mat.Zeros(1, hiddenSize)
	params["W2"] = W2.MulAll(weightInitStd)
	params["b2"] = mat.Zeros(1, hiddenSize)
}

func (tln *TwoLayerNet) predict(x *mat.Mat64) array.Array {
	W1 := tln.Params["W1"]
	b1 := tln.Params["b1"]
	W2 := tln.Params["W2"]
	b2 := tln.Params["b2"]
	a1 := x.Dot(W1).Add(b1)
	z1 := mat.Sigmoid(a1)
	a2 := z1.Dot(W2).Add(b2)
	y := array.Softmax(a2.Array)
	return y
}
