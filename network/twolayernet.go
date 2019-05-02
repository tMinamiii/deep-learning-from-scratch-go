package network

import (
	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
)

type TwoLayerNet struct {
	Params map[string]*mat.Matrix
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *TwoLayerNet {
	params := map[string]*mat.Matrix{}
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
func (tln *TwoLayerNet) Predict(x *mat.Matrix) *mat.Matrix {
	W1 := tln.Params["W1"]
	b1 := tln.Params["b1"]
	W2 := tln.Params["W2"]
	b2 := tln.Params["b2"]

	// start := time.Now()
	dota1 := x.Dot(W1)
	a1 := dota1.AddBroadCast(b1)
	z1 := mat.Sigmoid(a1)
	a2 := z1.Dot(W2).AddBroadCast(b2)
	y := mat.Softmax(a2)
	// end := time.Now()
	// fmt.Println(end.Sub(start))

	return y
}

func (tln *TwoLayerNet) Loss(x, t *mat.Matrix) float64 {
	y := tln.Predict(x)
	cee := mat.CrossEntropyError(y, t)
	return cee
}

func (tln *TwoLayerNet) Accuracy(x, t *mat.Matrix) float64 {
	y := tln.Predict(x)
	yMax := mat.ArgMax(y)
	tMax := mat.ArgMax(t)
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

func (tln *TwoLayerNet) NumericalGradient(x, t *mat.Matrix) map[string]*mat.Matrix {
	lossW := func(wArray array.Array) float64 {
		return tln.Loss(x, t)
	}
	grads := map[string]*mat.Matrix{}
	// fmt.Println("calc W1")
	grads["W1"] = mat.NumericalGradient(lossW, tln.Params["W1"])
	// fmt.Println("calc b1")
	grads["b1"] = mat.NumericalGradient(lossW, tln.Params["b1"])
	// fmt.Println("calc W2")
	grads["W2"] = mat.NumericalGradient(lossW, tln.Params["W2"])
	// fmt.Println("calc b2")
	grads["b2"] = mat.NumericalGradient(lossW, tln.Params["b2"])
	return grads
}
