package network

import (
	"github.com/naronA/zero_deeplearning/layer"
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type TwoLayerNet2 struct {
	Params  map[string]*mat.Matrix
	Affine1 *layer.Affine
	Relu    *layer.Relu
	Affine2 *layer.Affine
	Last    *layer.SoftmaxWithLoss
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewTwoLayerNet2(inputSize, hiddenSize, outputSize int, weightInitStd float64) *TwoLayerNet2 {
	params := map[string]*mat.Matrix{}

	W1, err := mat.NewRandnMatrix(inputSize, hiddenSize)
	if err != nil {
		panic(err)
	}
	W2, err := mat.NewRandnMatrix(hiddenSize, outputSize)
	if err != nil {
		panic(err)
	}
	params["W1"] = W1.Mul(weightInitStd)
	params["b1"] = mat.Zeros(1, hiddenSize)
	params["W2"] = W2.Mul(weightInitStd)
	params["b2"] = mat.Zeros(1, outputSize)
	aff1 := layer.NewAffine(params["W1"], params["b1"])
	relu := layer.NewRelu()
	aff2 := layer.NewAffine(params["W2"], params["b2"])
	last := layer.NewSfotmaxWithLoss()

	return &TwoLayerNet2{
		Params:  params,
		Affine1: aff1,
		Relu:    relu,
		Affine2: aff2,
		Last:    last,
	}
}

// Predict は、TwoLayerNetの精度計算をします
func (tln *TwoLayerNet2) Predict(x *mat.Matrix) *mat.Matrix {
	a1 := tln.Affine1.Forward(x)
	r := tln.Relu.Forward(a1)
	a2 := tln.Affine2.Forward(r)
	return a2
}

func (tln *TwoLayerNet2) Loss(x, t *mat.Matrix) float64 {
	y := tln.Predict(x)
	cee := tln.Last.Forward(y, t)
	return cee
}

func (tln *TwoLayerNet2) Accuracy(x, t *mat.Matrix) float64 {
	y := tln.Predict(x)
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

func (tln *TwoLayerNet2) Gradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	// forward
	tln.Loss(x, t)

	// backward
	dout := tln.Last.Backward(1.0)
	daff2 := tln.Affine2.Backward(dout)
	drelu := tln.Relu.Backward(daff2)
	tln.Affine1.Backward(drelu)

	grads := map[string]*mat.Matrix{}
	grads["W1"] = tln.Affine1.AdW
	grads["b1"] = tln.Affine1.Adb
	grads["W2"] = tln.Affine2.AdW
	grads["b2"] = tln.Affine2.Adb
	return grads
}

func (tln *TwoLayerNet2) NumericalGradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	lossW := func(wvec vec.Vector) float64 {
		return tln.Loss(x, t)
	}
	grads := map[string]*mat.Matrix{}
	grads["W1"] = mat.NumericalGradient(lossW, tln.Params["W1"])
	grads["b1"] = mat.NumericalGradient(lossW, tln.Params["b1"])
	grads["W2"] = mat.NumericalGradient(lossW, tln.Params["W2"])
	grads["b2"] = mat.NumericalGradient(lossW, tln.Params["b2"])
	return grads
}
