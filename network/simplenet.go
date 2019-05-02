package network

import (
	"github.com/naronA/zero_deeplearning/mat"
)

type SimpleNet struct {
	W *mat.Mat64
}

func NewSimpleNet(weight *mat.Mat64) *SimpleNet {
	return &SimpleNet{
		W: weight,
	}
}

func (sn *SimpleNet) Predict(x *mat.Mat64) *mat.Mat64 {
	return x.Dot(sn.W)
}

func (sn *SimpleNet) Loss(x, t *mat.Mat64) float64 {
	z := sn.Predict(x)
	y := mat.Softmax(z)
	return mat.CrossEntropyError(y, t)
}
