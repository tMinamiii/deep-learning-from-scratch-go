package network

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/layer"
	"github.com/naronA/zero_deeplearning/mat"
)

type TwoLayerNet struct {
	Params    map[string]*mat.Matrix
	Layers    map[string]layer.Layer
	Sequence  []string
	LastLayer *layer.SoftmaxWithLoss
	AffineNum int
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *TwoLayerNet {
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
	params["W1"] = W1.Mul(weightInitStd)
	params["b1"] = mat.Zeros(1, hiddenSize)
	params["W2"] = W2.Mul(weightInitStd)
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
	}
}

// Predict は、TwoLayerNetの精度計算をします
func (tln *TwoLayerNet) Predict(x *mat.Matrix) *mat.Matrix {
	for _, k := range tln.Sequence {
		x = tln.Layers[k].Forward(x)
	}
	return x
}

func (tln *TwoLayerNet) Loss(x, t *mat.Matrix) float64 {
	y := tln.Predict(x)
	cee := tln.LastLayer.Forward(y, t)
	return cee
}

func (tln *TwoLayerNet) Accuracy(x, t *mat.Matrix) float64 {
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

func (tln *TwoLayerNet) Gradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	// forward
	tln.Loss(x, t)

	// backward
	dout := tln.LastLayer.Backward(1.0)

	for i := len(tln.Sequence) - 1; i >= 0; i-- {
		key := tln.Sequence[i]
		dout = tln.Layers[key].Backward(dout)
	}

	grads := map[string]*mat.Matrix{}

	for i := 1; i <= tln.AffineNum; i++ {
		l := fmt.Sprintf("Affine%d", i)
		w := fmt.Sprintf("W%d", i)
		b := fmt.Sprintf("b%d", i)
		if v, ok := tln.Layers[l].(*layer.Affine); ok {
			grads[w] = v.DW
			grads[b] = v.DB
		}
	}
	return grads
}

func (tln *TwoLayerNet) UpdateParams(params map[string]*mat.Matrix) {
	tln.Params = params

	for i := 1; i <= tln.AffineNum; i++ {
		l := fmt.Sprintf("Affine%d", i)
		w := fmt.Sprintf("W%d", i)
		b := fmt.Sprintf("b%d", i)
		if v, ok := tln.Layers[l].(*layer.Affine); ok {
			v.W = params[w]
			v.B = params[b]

		}
	}
}
