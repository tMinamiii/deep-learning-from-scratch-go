package network

import (
	"fmt"
	"math"

	"github.com/naronA/zero_deeplearning/layer"
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/optimizer"
)

type TwoLayerNet struct {
	Params    map[string]*mat.Matrix
	Layers    map[string]layer.Layer
	Sequence  []string
	LastLayer *layer.SoftmaxWithLoss
	AffineNum int
	Optimizer optimizer.Optimizer
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewTwoLayerNet(opt optimizer.Optimizer, inputSize, hiddenSize, outputSize int) *TwoLayerNet {
	params := map[string]*mat.Matrix{}
	layers := map[string]layer.Layer{}

	W1, err := mat.NewRandnMatrix(inputSize, hiddenSize)
	if err != nil {
		panic(err)
	}
	W2, err := mat.NewRandnMatrix(hiddenSize, outputSize)
	if err != nil {
		panic(err)
	}
	params["W1"] = mat.Div(W1, math.Sqrt(2.0*float64(inputSize))) // weightInitStd
	params["b1"] = mat.Zeros(1, hiddenSize)
	params["W2"] = mat.Div(W2, math.Sqrt(2.0*float64(hiddenSize)))
	params["b2"] = mat.Zeros(1, outputSize)
	layers["Affine1"] = layer.NewAffine(params["W1"], params["b1"])
	layers["Relu1"] = layer.NewRelu()
	layers["Affine2"] = layer.NewAffine(params["W2"], params["b2"])
	seq := []string{"Affine1", "Relu1", "Affine2"}
	last := layer.NewSfotmaxWithLoss()

	return &TwoLayerNet{
		Params:    params,
		Layers:    layers,
		LastLayer: last,
		Sequence:  seq,
		AffineNum: 2,
		Optimizer: opt,
	}
}

// Predict は、TwoLayerNetの精度計算をします
func (net *TwoLayerNet) Predict(x *mat.Matrix) *mat.Matrix {
	for _, k := range net.Sequence {
		x = net.Layers[k].Forward(x)
	}
	return x
}

func (net *TwoLayerNet) Loss(x, t *mat.Matrix) float64 {
	y := net.Predict(x)
	cee := net.LastLayer.Forward(y, t)
	return cee
}

func (net *TwoLayerNet) Accuracy(x, t *mat.Matrix) float64 {
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

func (net *TwoLayerNet) Gradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	// forward
	net.Loss(x, t)

	// backward
	dout := net.LastLayer.Backward(1.0)

	for i := len(net.Sequence) - 1; i >= 0; i-- {
		key := net.Sequence[i]
		dout = net.Layers[key].Backward(dout)
	}

	grads := map[string]*mat.Matrix{}

	for i := 1; i <= net.AffineNum; i++ {
		l := fmt.Sprintf("Affine%d", i)
		w := fmt.Sprintf("W%d", i)
		b := fmt.Sprintf("b%d", i)
		if v, ok := net.Layers[l].(*layer.Affine); ok {
			grads[w] = v.DW
			grads[b] = v.DB
		}
	}
	return grads
}

func (net *TwoLayerNet) UpdateParams(grads map[string]*mat.Matrix) {
	net.Params = net.Optimizer.Update(net.Params, grads)

	for i := 1; i <= net.AffineNum; i++ {
		l := fmt.Sprintf("Affine%d", i)
		w := fmt.Sprintf("W%d", i)
		b := fmt.Sprintf("b%d", i)
		if v, ok := net.Layers[l].(*layer.Affine); ok {
			v.W = net.Params[w]
			v.B = net.Params[b]
		}
	}
}
