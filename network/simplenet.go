package network

import (
	"github.com/naronA/zero_deeplearning/mat"
)

type SimpleNet struct {
	W *mat.Matrix
}

func NewSimpleNet(weight *mat.Matrix) *SimpleNet {
	return &SimpleNet{
		W: weight,
	}
}

func (net *SimpleNet) Predict(x *mat.Matrix) *mat.Matrix {
	return mat.Dot(x, net.W)
}

func (net *SimpleNet) Loss(x, t *mat.Matrix) float64 {
	z := net.Predict(x)
	y := mat.Softmax(z)
	return mat.CrossEntropyError(y, t)
}
