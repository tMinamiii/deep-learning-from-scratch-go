package network

import (
	"fmt"
	"math"

	"github.com/naronA/zero_deeplearning/layer"
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/optimizer"
)

type MultiLayerNet struct {
	Params            map[string]*mat.Matrix
	Layers            map[string]layer.Layer
	Sequence          []string
	LastLayer         *layer.SoftmaxWithLoss
	Optimizer         optimizer.Optimizer
	HiddenLayerNum    int
	WeightDecayLambda float64
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewTwoLayerNet(
	opt optimizer.Optimizer,
	inputSize int,
	hiddenSize int,
	outputSize int,
	weightDeceyLambda float64) *MultiLayerNet {
	params := map[string]*mat.Matrix{}
	layers := map[string]layer.Layer{}

	W1, err := mat.NewRandnMatrix(inputSize, hiddenSize)
	if err != nil {
		panic(err)
	}
	W2, err := mat.NewRandnMatrix(hiddenSize, hiddenSize)
	if err != nil {
		panic(err)
	}
	W3, err := mat.NewRandnMatrix(hiddenSize, outputSize)
	if err != nil {
		panic(err)
	}

	params["W1"] = mat.Div(W1, math.Sqrt(2.0*float64(inputSize))) // weightInitStd
	params["b1"] = mat.Zeros(1, hiddenSize)
	params["W2"] = mat.Div(W2, math.Sqrt(2.0*float64(hiddenSize)))
	params["b2"] = mat.Zeros(1, hiddenSize)
	params["W3"] = mat.Div(W3, math.Sqrt(2.0*float64(hiddenSize)))
	params["b3"] = mat.Zeros(1, outputSize)

	layers["Affine1"] = layer.NewAffine(params["W1"], params["b1"])
	layers["BatchNorm1"] = layer.NewBatchNorimalization(1.0, 0.0)
	layers["Relu1"] = layer.NewRelu()
	layers["Dropout1"] = layer.NewDropout(0.5)

	layers["Affine2"] = layer.NewAffine(params["W2"], params["b2"])
	layers["BatchNorm2"] = layer.NewBatchNorimalization(1.0, 0.0)
	layers["Relu2"] = layer.NewRelu()
	layers["Dropout2"] = layer.NewDropout(0.5)

	layers["Affine3"] = layer.NewAffine(params["W3"], params["b3"])

	seq := []string{
		"Affine1",
		"BatchNorm1",
		"Relu1",
		"Affine2",
		"BatchNorm2",
		"Relu2",
		"Affine3",
	}

	last := layer.NewSfotmaxWithLoss()

	return &MultiLayerNet{
		Params:            params,
		Layers:            layers,
		LastLayer:         last,
		Sequence:          seq,
		Optimizer:         opt,
		HiddenLayerNum:    2,
		WeightDecayLambda: weightDeceyLambda,
	}
}

// Predict は、TwoLayerNetの精度計算をします
func (net *MultiLayerNet) Predict(x *mat.Matrix, trainFlg bool) *mat.Matrix {
	for _, k := range net.Sequence {
		x = net.Layers[k].Forward(x, trainFlg)
	}
	return x
}

func (net *MultiLayerNet) Loss(x, t *mat.Matrix, trainFlg bool) float64 {
	y := net.Predict(x, trainFlg)

	weightDecey := 0.0
	for i := 1; i < net.HiddenLayerNum+2; i++ {
		k := fmt.Sprintf("W%d", i)
		W := net.Params[k]
		weightDecey += 0.5 * net.WeightDecayLambda * mat.SumAll(mat.Pow(W, 2))
	}
	cee := net.LastLayer.Forward(y, t) + weightDecey
	return cee
}

func (net *MultiLayerNet) Accuracy(x, t *mat.Matrix) float64 {
	y := net.Predict(x, false)
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

func (net *MultiLayerNet) Gradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	// forward
	net.Loss(x, t, true)

	// backward
	dout := net.LastLayer.Backward(1.0)

	for i := len(net.Sequence) - 1; i >= 0; i-- {
		key := net.Sequence[i]
		dout = net.Layers[key].Backward(dout)
	}

	grads := map[string]*mat.Matrix{}

	for i := 1; i < net.HiddenLayerNum+2; i++ {
		l := fmt.Sprintf("Affine%d", i)
		w := fmt.Sprintf("W%d", i)
		b := fmt.Sprintf("b%d", i)
		if v, ok := net.Layers[l].(*layer.Affine); ok {
			grads[w] = mat.Add(v.DW, mat.Mul(net.WeightDecayLambda, v.W))
			grads[b] = v.DB
		}
	}
	return grads
}

func (net *MultiLayerNet) UpdateParams(grads map[string]*mat.Matrix) {
	net.Params = net.Optimizer.Update(net.Params, grads)

	for i := 1; i < net.HiddenLayerNum+2; i++ {
		l := fmt.Sprintf("Affine%d", i)
		w := fmt.Sprintf("W%d", i)
		b := fmt.Sprintf("b%d", i)
		if v, ok := net.Layers[l].(*layer.Affine); ok {
			v.W = net.Params[w]
			v.B = net.Params[b]
		}
	}
}
