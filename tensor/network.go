package tensor

import (
	"fmt"
	"time"

	"github.com/naronA/zero_deeplearning/vec"
)

type SimpleConvNet struct {
	Params    map[string]*Tensor
	Layers    map[string]Layer
	Sequence  []string
	LastLayer *SoftmaxWithLoss
	Optimizer Optimizer
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
	opt Optimizer,
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

	W1Rnd := NewRandnT4D(filterNum, inputDim.Channel, filterSize, filterSize)
	W2Rnd := NewRandnMatrix(poolOutputSize, hiddenSize)
	W3Rnd := NewRandnMatrix(hiddenSize, outputSize)
	params := map[string]*Tensor{}
	// t4dparams := map[string]num.Tensor4D{}

	W1 := Mul(W1Rnd, &Tensor{Val: weightInitStd})
	b1 := Zeros([]int{1, filterNum})
	W2 := Mul(W2Rnd, &Tensor{Val: weightInitStd})
	b2 := Zeros([]int{1, hiddenSize})
	W3 := Mul(W3Rnd, &Tensor{Val: weightInitStd})
	b3 := Zeros([]int{1, outputSize})

	params["W1"] = W1
	params["b1"] = b1
	params["W2"] = W2
	params["b2"] = b2
	params["W3"] = W3
	params["b3"] = b3

	layers := map[string]Layer{}
	layers["Conv1"] = NewConvolution(W1, b1, convParams.Stride, convParams.Pad)
	layers["Relu1"] = NewRelu()
	layers["Pool1"] = NewPooling(2, 2, 2, 0)
	layers["Affine1"] = NewAffine(W2, b2)
	layers["Relu2"] = NewRelu()
	layers["Affine2"] = NewAffine(W3, b3)

	seq := []string{
		"Conv1",
		"Relu1",
		"Pool1",
		"Affine1",
		"Relu2",
		"Affine2",
	}

	last := NewSfotmaxWithLoss()

	return &SimpleConvNet{
		Params:    params,
		Layers:    layers,
		LastLayer: last,
		Sequence:  seq,
		Optimizer: opt,
	}
}

func (net *SimpleConvNet) Predict(x *Tensor) *Tensor {
	for _, k := range net.Sequence {
		x = net.Layers[k].Forward(x)
	}
	return x
}

func (net *SimpleConvNet) Loss(x *Tensor, t *Tensor) float64 {
	if x == nil || t == nil {
		fmt.Println(x, t)
	}
	y := net.Predict(x)
	return net.LastLayer.Forward(y, t)
}

func (net *SimpleConvNet) Gradient(x, t *Tensor) map[string]*Tensor {
	// forward
	net.Loss(x, t)
	var dout *Tensor = net.LastLayer.Backward()

	for i := len(net.Sequence) - 1; i >= 0; i-- {
		key := net.Sequence[i]
		dout = net.Layers[key].Backward(dout)
	}

	grads := map[string]*Tensor{}
	grads["W1"] = net.Layers["Conv1"].(*Convolution).DW
	grads["b1"] = net.Layers["Conv1"].(*Convolution).DB
	grads["W2"] = net.Layers["Affine1"].(*Affine).DW
	grads["b2"] = net.Layers["Affine1"].(*Affine).DB
	grads["W3"] = net.Layers["Affine2"].(*Affine).DW
	grads["b3"] = net.Layers["Affine2"].(*Affine).DB
	return grads

}

func (net *SimpleConvNet) UpdateParams(grads map[string]*Tensor) {
	net.Params = net.Optimizer.Update(net.Params, grads)

	conv1 := net.Layers["Conv1"].(*Convolution)
	conv1.W = net.Params["W1"]
	conv1.B = net.Params["b1"]

	affine1 := net.Layers["Affine1"].(*Affine)
	affine1.W = net.Params["W2"]
	affine1.B = net.Params["b2"]

	affine2 := net.Layers["Affine2"].(*Affine)
	affine2.W = net.Params["W3"]
	affine2.B = net.Params["b3"]
}

func (net *SimpleConvNet) Accuracy(x, t *Tensor) float64 {
	var sem = make(chan struct{}, 40)
	accuracy := 0.0
	size := 5
	count := 0
	ch := make(chan float64)
	start := time.Now()
	for i := 0; i < len(x.T4D); i += size {
		count++
		train := x.T4D[i : i+size]
		v := make(vec.Vector, 0, t.Mat.Rows*size)
		for k := i; k < i+size; k++ {
			if k >= t.Mat.Rows {
				break
			}
			v = append(v, t.Mat.SliceRow(k)...)
		}
		a, b, c, d := train.Shape()
		tt := &Tensor{
			T4D:   train,
			Shape: []int{a, b, c, d},
		}
		test := &Tensor{
			Mat: &Matrix{
				Vector:  v,
				Rows:    size,
				Columns: t.Mat.Columns,
			},
			Shape: []int{size, t.Mat.Columns},
		}
		sem <- struct{}{}
		go calcAcc(net, tt, test, ch)
		<-sem
	}
	for i := 0; i < len(x.T4D)/size; i++ {
		accuracy += <-ch
	}
	close(ch)
	end := time.Now()
	fmt.Printf("elapstime = %v accuracy %f\n", end.Sub(start), accuracy/float64(count))
	return accuracy / float64(count)
}

func calcAcc(net *SimpleConvNet, train, test *Tensor, ch chan float64) {
	y := net.Predict(train)
	yMax := y.ArgMax(1)
	tMax := test.ArgMax(1)
	sum := 0.0
	r := y.Shape[0]
	for i, v := range yMax {
		if v == tMax[i] {
			sum += 1.0
		}
	}
	ch <- sum / float64(r)
}
