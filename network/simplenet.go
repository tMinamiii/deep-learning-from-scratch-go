package network

import (
	"github.com/naronA/zero_deeplearning/num"
)

type SimpleNet struct {
	W num.Matrix
}

func NewSimpleNet(weight num.Matrix) *SimpleNet {
	return &SimpleNet{
		W: weight,
	}
}

func (net *SimpleNet) Predict(x num.Matrix) num.Matrix {
	return num.Dot(x, net.W)
}

func (net *SimpleNet) Loss(x, t num.Matrix) float64 {
	z := net.Predict(x)
	y := num.Softmax(z)
	return num.CrossEntropyError(y, t)
}
