package optimizer

import (
	"math"

	"github.com/naronA/zero_deeplearning/num"
)

type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Iter  int
	M     map[string]num.Matrix
	V     map[string]num.Matrix
}

func NewAdam(lr float64) *Adam {
	return &Adam{
		LR:    lr,
		Beta1: 0.9,
		Beta2: 0.999,
		Iter:  0,
		M:     nil,
		V:     nil,
	}
}

func (a *Adam) Update(params, grads map[string]num.Matrix) map[string]num.Matrix {
	if a.M == nil {
		a.M = map[string]num.Matrix{}
		a.V = map[string]num.Matrix{}
		for k, v := range params {
			a.M[k] = num.ZerosLike(v)
			a.V[k] = num.ZerosLike(v)
		}
	}
	a.Iter++
	fIter := float64(a.Iter)
	lrT := a.LR * math.Sqrt(1.0-math.Pow(a.Beta2, fIter)) / (1.0 - math.Pow(a.Beta1, fIter))

	newParams := map[string]num.Matrix{}
	for k, g := range grads {
		a.M[k] = num.Add(a.M[k], num.Mul(1.0-a.Beta1, num.Sub(g, a.M[k])))
		a.V[k] = num.Add(a.V[k], num.Mul(1.0-a.Beta2, num.Sub(num.Pow(g, 2), a.V[k])))

		delta := num.Div(num.Mul(lrT, a.M[k]), num.Add(num.Sqrt(a.V[k]), 1e-7))
		newParams[k] = num.Sub(params[k], delta)
	}
	return newParams
}
