package network

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/layer"
	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/optimizer"
	"github.com/naronA/zero_deeplearning/vec"
)

type SimpleConvNet struct {
	Params map[string]interface{}
	// T4DParams      map[string]num.Tensor4D
	T4DLayers map[string]layer.T4DLayer
	Sequence  []string
	LastLayer *layer.SoftmaxWithLoss
	Optimizer optimizer.AnyOptimizer
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
	opt optimizer.AnyOptimizer,
	inputDim *InputDim,
	convParams *ConvParams,
	hiddenSize int,
	outputSize int,
	weightInitStd float64,
) *SimpleConvNet {
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
	params := map[string]interface{}{}
	// t4dparams := map[string]num.Tensor4D{}

	W1 := num.MulT4D(W1Rnd, weightInitStd)
	b1 := num.Zeros(1, filterNum)
	W2 := num.Mul(W2Rnd, weightInitStd)
	b2 := num.Zeros(1, hiddenSize)
	W3 := num.Mul(W3Rnd, weightInitStd)
	b3 := num.Zeros(1, outputSize)

	params["W1"] = W1
	params["b1"] = b1
	params["W2"] = W2
	params["b2"] = b2
	params["W3"] = W3
	params["b3"] = b3

	layers := map[string]layer.T4DLayer{}
	layers["Conv1"] = layer.NewConvolution(W1, b1, convParams.Stride, convParams.Pad)
	layers["Relu1"] = layer.NewReluT4D()
	layers["Pool1"] = layer.NewPooling(2, 2, 2, 0)
	layers["Affine1"] = layer.NewAffineT4D(W2, b2)
	layers["Relu2"] = layer.NewReluT4D()
	layers["Affine2"] = layer.NewAffineT4D(W3, b3)

	seq := []string{
		"Conv1",
		"Relu1",
		"Pool1",
		"Affine1",
		"Relu2",
		"Affine2",
	}

	last := layer.NewSfotmaxWithLoss()

	return &SimpleConvNet{
		Params: params,
		// T4DParams:      t4dparams,
		T4DLayers: layers,
		LastLayer: last,
		Sequence:  seq,
		Optimizer: opt,
		// WeightDecayLambda: weightDeceyLambda,
	}
}

func (net *SimpleConvNet) Predict(x interface{}) interface{} {
	for _, k := range net.Sequence {
		x = net.T4DLayers[k].Forward(x)
	}
	return x
}

func (net *SimpleConvNet) Loss(x num.Tensor4D, t *num.Matrix) float64 {
	if x == nil || t == nil {
		fmt.Println(x, t)
	}
	y := net.Predict(x).(*num.Matrix)
	return net.LastLayer.Forward(y, t)
}

func (net *SimpleConvNet) Gradient(x num.Tensor4D, t *num.Matrix) map[string]interface{} {
	// forward
	net.Loss(x, t)
	var dout interface{} = net.LastLayer.Backward(0)

	for i := len(net.Sequence) - 1; i >= 0; i-- {
		key := net.Sequence[i]
		dout = net.T4DLayers[key].Backward(dout)
	}

	grads := map[string]interface{}{}
	grads["W1"] = net.T4DLayers["Conv1"].(*layer.Convolution).DW
	grads["b1"] = net.T4DLayers["Conv1"].(*layer.Convolution).DB
	grads["W2"] = net.T4DLayers["Affine1"].(*layer.AffineT4D).DW
	grads["b2"] = net.T4DLayers["Affine1"].(*layer.AffineT4D).DB
	grads["W3"] = net.T4DLayers["Affine2"].(*layer.AffineT4D).DW
	grads["b3"] = net.T4DLayers["Affine2"].(*layer.AffineT4D).DB
	return grads

}

func (net *SimpleConvNet) UpdateParams(grads map[string]interface{}) {
	net.Params = net.Optimizer.Update(net.Params, grads)

	conv1 := net.T4DLayers["Conv1"].(*layer.Convolution)
	conv1.W = net.Params["W1"].(num.Tensor4D)
	conv1.B = net.Params["b1"].(*num.Matrix)

	affine1 := net.T4DLayers["Affine1"].(*layer.AffineT4D)
	affine2 := net.T4DLayers["Affine2"].(*layer.AffineT4D)
	affine1.W = net.Params["W2"].(*num.Matrix)
	affine1.B = net.Params["b2"].(*num.Matrix)
	affine2.W = net.Params["W3"].(*num.Matrix)
	affine2.B = net.Params["b3"].(*num.Matrix)
}

func (net *SimpleConvNet) Accuracy(x num.Tensor4D, t *num.Matrix) float64 {
	accuracy := 0.0
	miniBatchSize := 5000
	count := 0
	fmt.Println(len(x))
	for i := 0; i < len(x); i += miniBatchSize {
		count++
		y := net.Predict(x[i : i+miniBatchSize]).(*num.Matrix)
		yMax := num.ArgMax(y, 1)

		v := vec.Vector{}
		for j := i; j < i+miniBatchSize; j++ {
			if j >= t.Rows {
				break
			}
			v = append(v, t.SliceRow(j)...)
		}
		test := &num.Matrix{
			Vector:  v,
			Rows:    miniBatchSize,
			Columns: t.Columns,
		}

		tMax := num.ArgMax(test, 1)
		sum := 0.0
		r, _ := y.Shape()
		for i, v := range yMax {
			if v == tMax[i] {
				sum += 1.0
			}
		}
		accuracy += sum / float64(r)
		fmt.Printf("temp %f\n", accuracy)
	}
	fmt.Printf("accuracy %f\n", accuracy)

	return accuracy / float64(count)
}
