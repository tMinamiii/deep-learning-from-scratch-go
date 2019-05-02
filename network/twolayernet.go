package network

import (
	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
)

type TwoLayerNet struct {
	Params map[string]*mat.Mat64
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *TwoLayerNet {
	params := map[string]*mat.Mat64{}
	W1, err := mat.NewRandnMat64(inputSize, hiddenSize)
	if err != nil {
		panic(err)
	}
	W2, err := mat.NewRandnMat64(hiddenSize, outputSize)
	if err != nil {
		panic(err)
	}
	params["W1"] = W1.MulAll(weightInitStd)
	params["b1"] = mat.Zeros(1, hiddenSize)
	params["W2"] = W2.MulAll(weightInitStd)
	params["b2"] = mat.Zeros(1, outputSize)
	return &TwoLayerNet{Params: params}
}

// Predict は、TwoLayerNetの精度計算をします
func (tln *TwoLayerNet) Predict(x *mat.Mat64) *mat.Mat64 {
	W1 := tln.Params["W1"]
	b1 := tln.Params["b1"]
	W2 := tln.Params["W2"]
	b2 := tln.Params["b2"]

	a1 := x.Dot(W1).AddBroadCast(b1)
	z1 := mat.Sigmoid(a1)
	a2 := z1.Dot(W2).AddBroadCast(b2)
	y := mat.Softmax(a2)
	return y
}

func (tln *TwoLayerNet) Loss(x, t *mat.Mat64) float64 {
	y := tln.Predict(x)
	return mat.CrossEntropyError(y, t)
}

func (tln *TwoLayerNet) Accuracy(x, t *mat.Mat64) float64 {
	y := tln.Predict(x)
	yMax := mat.ArgMax(y)
	tMax := mat.ArgMax(t)
	sum := 0.0
	r, _ := x.Shape()
	for i, v := range yMax {
		if v == tMax[i] {
			sum += float64(v)
		}
	}
	accuracy := sum / float64(r)
	return accuracy
}

func (tln *TwoLayerNet) NumericalGradient(x, t *mat.Mat64) map[string]*mat.Mat64 {
	lossW := func(wArray array.Array) float64 {
		return tln.Loss(x, t)
	}
	grads := map[string]*mat.Mat64{}
	grads["W1"] = mat.NumericalGradient(lossW, tln.Params["W1"])
	grads["b1"] = mat.NumericalGradient(lossW, tln.Params["b1"])
	grads["W2"] = mat.NumericalGradient(lossW, tln.Params["W2"])
	grads["b2"] = mat.NumericalGradient(lossW, tln.Params["b2"])
	return grads
}
