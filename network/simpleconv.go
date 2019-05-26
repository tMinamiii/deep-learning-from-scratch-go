package network

import (
	"strings"

	"github.com/naronA/zero_deeplearning/layer"
	"github.com/naronA/zero_deeplearning/num"
)

type SimpleConvNet struct {
	Params    map[string]*num.Matrix
	T4DParams map[string]num.Tensor4D
	Layers    map[string]layer.Layer
	T4DLayers map[string]layer.T4DLayer
	Sequence  []string
	LastLayer *layer.SoftmaxWithLoss
	// Optimizer         optimizer.Optimizer
	// HiddenLayerNum    int
	// WeightDecayLambda float64
}

type InputDim struct {
	Channel int
	Height  int
	Weidth  int
}

type ConvParams struct {
	FilterNum  int
	FilterSize int
	Pad        int
	Stride     int
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewSimpleConvNet(
	inputDim InputDim, convParams ConvParams, hiddenSize int, outputSize int, weightInitStd float64) *SimpleConvNet {
	filterNum := convParams.FilterNum
	filterSize := convParams.FilterSize
	filterPad := convParams.Pad
	filterStride := convParams.Stride

	inputSize := inputDim.Height
	convOutputSize := (inputSize-filterSize+2*filterPad)/filterStride + 1
	poolOutputSize := filterNum * (convOutputSize / 2) * (convOutputSize / 2)

	W1Rnd, err := num.NewRandnT4D(filterNum, inputDim.Channel, filterSize, filterSize)
	if err != nil {
		panic(err)
	}
	W2Rnd, err := num.NewRandnMatrix(poolOutputSize, hiddenSize)
	if err != nil {
		panic(err)
	}
	W3Rnd, err := num.NewRandnMatrix(hiddenSize, outputSize)
	if err != nil {
		panic(err)
	}
	params := map[string]*num.Matrix{}
	t4dparams := map[string]num.Tensor4D{}

	W1 := num.MulT4D(W1Rnd, weightInitStd)
	b1 := num.Zeros(1, filterNum)
	W2 := num.Mul(W2Rnd, weightInitStd)
	b2 := num.Zeros(1, hiddenSize)
	W3 := num.Mul(W3Rnd, weightInitStd)
	b3 := num.Zeros(1, outputSize)

	t4dparams["W1"] = num.MulT4D(W1, weightInitStd)
	params["b1"] = num.Zeros(1, filterNum)
	params["W2"] = num.Mul(W2, weightInitStd)
	params["b2"] = num.Zeros(1, hiddenSize)
	params["W3"] = num.Mul(W3, weightInitStd)
	params["b3"] = num.Zeros(1, outputSize)

	layers := map[string]layer.Layer{}
	layersT4d := map[string]layer.T4DLayer{}
	layersT4d["Conv1"] = layer.NewConvolution(W1, b1, convParams.Stride, convParams.Pad)
	layers["Relu1"] = layer.NewRelu()
	layersT4d["Pool1"] = layer.NewPooling(2, 2, 2, 0)
	layers["Affine1"] = layer.NewAffine(W2, b2)
	layers["Relu2"] = layer.NewRelu()
	layers["Affine2"] = layer.NewAffine(W3, b3)

	seq := []string{
		"Conv1",
		"Relu1",
		"Pool1",
		"Affine1",
		"Relu2",
		"Affile2",
	}

	last := layer.NewSfotmaxWithLoss()

	return &SimpleConvNet{
		Params:    params,
		T4DParams: t4dparams,
		Layers:    layers,
		T4DLayers: layersT4d,
		LastLayer: last,
		Sequence:  seq,
		// Optimizer:         opt,
		// HiddenLayerNum:    1,
		// WeightDecayLambda: weightDeceyLambda,
	}
}

func (net *SimpleConvNet) Predict(x num.Tensor4D, trainFlg bool) num.Tensor4D {
	for _, k := range net.Sequence {
		if strings.HasPrefix(k, "Conv") {
			x = net.T4DLayers[k].Forward(x)
			return x
		}

		t4d := num.ZerosLikeT4D(x)
		for i, t3d := range x {
			for j, mat := range t3d {
				t4d[i][j] = net.Layers[k].Forward(mat, trainFlg)
			}
		}
		x = t4d
	}
	return x
}
