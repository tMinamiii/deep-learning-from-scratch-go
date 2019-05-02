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

func (sn *SimpleNet) Predict(x *mat.Matrix) *mat.Matrix {
	return x.Dot(sn.W)
}

func (sn *SimpleNet) Loss(x, t *mat.Matrix) float64 {
	z := sn.Predict(x)
	y := mat.Softmax(z)
	return mat.CrossEntropyError(y, t)
}
