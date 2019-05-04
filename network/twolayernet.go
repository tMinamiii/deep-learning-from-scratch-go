package network

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type TwoLayerNet struct {
	Params map[string]*mat.Matrix
}

// NewTwoLayerNet は、TwoLayerNetのコンストラクタ
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *TwoLayerNet {
	params := map[string]*mat.Matrix{}
	W1, err := mat.NewRandnMatrix(inputSize, hiddenSize)
	if err != nil {
		panic(err)
	}
	W2, err := mat.NewRandnMatrix(hiddenSize, outputSize)
	if err != nil {
		panic(err)
	}
	params["W1"], _ = W1.Mul(weightInitStd)
	params["b1"] = mat.Zeros(1, hiddenSize)
	params["W2"], _ = W2.Mul(weightInitStd)
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
	dota1 := mat.Dot(x, W1)
	a1, _ := dota1.Add(b1.Vector)
	z1 := mat.Sigmoid(a1)
	a2, _ := mat.Dot(z1, W2).Add(b2.Vector)
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

func (tln *TwoLayerNet) NumericalGradient(x, t *mat.Matrix) map[string]*mat.Matrix {
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
