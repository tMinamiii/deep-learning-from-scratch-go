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

func (self *SimpleNet) Predict(x *mat.Matrix) *mat.Matrix {
	return mat.Dot(x, self.W)
}

func (self *SimpleNet) Loss(x, t *mat.Matrix) float64 {
	z := self.Predict(x)
	y := mat.Softmax(z)
	return mat.CrossEntropyError(y, t)
}
