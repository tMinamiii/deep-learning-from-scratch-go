package network

import (
	"github.com/naronA/zero_deeplearning/layer"
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type LayeredTwoLayerNet struct {
	Params    map[string]*mat.Matrix
	Affine1   *layer.Affine
	Relu1     *layer.Relu
	Sigmoid1  *layer.Sigmoid
	Affine2   *layer.Affine
	LastLayer *layer.SoftmaxWithLoss
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewLayeredTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *LayeredTwoLayerNet {
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
	sig := layer.NewSigmoid()
	aff2 := layer.NewAffine(params["W2"], params["b2"])
	last := layer.NewSfotmaxWithLoss()

	return &LayeredTwoLayerNet{
		Params:    params,
		Affine1:   aff1,
		Relu1:     relu,
		Sigmoid1:  sig,
		Affine2:   aff2,
		LastLayer: last,
	}
}

// Predict は、TwoLayerNetの精度計算をします
func (tln *LayeredTwoLayerNet) Predict(x *mat.Matrix) *mat.Matrix {
	f := tln.Affine1.Forward(x)
	// f = tln.Relu1.Forward(f)
	f = tln.Sigmoid1.Forward(f)
	f = tln.Affine2.Forward(f)
	return f
}

func (tln *LayeredTwoLayerNet) Loss(x, t *mat.Matrix) float64 {
	y := tln.Predict(x)
	cee := tln.LastLayer.Forward(y, t)
	return cee
}

func (tln *LayeredTwoLayerNet) Accuracy(x, t *mat.Matrix) float64 {
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

func (tln *LayeredTwoLayerNet) Gradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	// forward
	tln.Loss(x, t)

	// backward
	dout := tln.LastLayer.Backward(1.0)
	dout = tln.Affine2.Backward(dout)
	// dout = tln.Relu1.Backward(dout)
	dout = tln.Sigmoid1.Backward(dout)
	tln.Affine1.Backward(dout)

	grads := map[string]*mat.Matrix{}
	grads["W1"] = tln.Affine1.DW
	grads["b1"] = tln.Affine1.DB
	grads["W2"] = tln.Affine2.DW
	grads["b2"] = tln.Affine2.DB
	return grads
}

func (tln *LayeredTwoLayerNet) UpdateParams(params map[string]*mat.Matrix) {
	tln.Params = params
	tln.Affine1.W = params["W1"]
	tln.Affine1.B = params["b1"]
	tln.Affine2.W = params["W2"]
	tln.Affine2.B = params["b2"]
}

func (tln *LayeredTwoLayerNet) NumericalGradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	lossW := func(wvec vec.Vector) float64 {
		return tln.LossOld(x, t)
	}
	grads := map[string]*mat.Matrix{}
	grads["W1"] = mat.NumericalGradient(lossW, tln.Params["W1"])
	grads["b1"] = mat.NumericalGradient(lossW, tln.Params["b1"])
	grads["W2"] = mat.NumericalGradient(lossW, tln.Params["W2"])
	grads["b2"] = mat.NumericalGradient(lossW, tln.Params["b2"])
	return grads
}

// Predict は、TwoLayerNetの精度計算をします
func (tln *LayeredTwoLayerNet) PredictOld(x *mat.Matrix) *mat.Matrix {
	W1 := tln.Params["W1"]
	b1 := tln.Params["b1"]
	W2 := tln.Params["W2"]
	b2 := tln.Params["b2"]

	dota1 := mat.Dot(x, W1)
	a1 := dota1.Add(b1)
	z1 := mat.Sigmoid(a1)
	a2 := mat.Dot(z1, W2).Add(b2)
	y := mat.Softmax(a2)

	return y
}

func (tln *LayeredTwoLayerNet) LossOld(x, t *mat.Matrix) float64 {
	y := tln.PredictOld(x)
	cee := mat.CrossEntropyError(y, t)
	return cee
}
