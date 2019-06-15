package tensor

import (
	"math"
)

type Optimizer interface {
	Update(map[string]*Tensor, map[string]*Tensor) map[string]*Tensor
}

type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Iter  int
	M     map[string]*Tensor
	V     map[string]*Tensor
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

func (a *Adam) Update(params, grads map[string]*Tensor) map[string]*Tensor {
	if a.M == nil {
		a.M = map[string]*Tensor{}
		a.V = map[string]*Tensor{}
		for k, v := range params {
			a.M[k] = ZerosLike(v)
			a.V[k] = ZerosLike(v)
		}
	}
	a.Iter++
	fIter := float64(a.Iter)
	lrT := a.LR * math.Sqrt(1.0-math.Pow(a.Beta2, fIter)) / (1.0 - math.Pow(a.Beta1, fIter))

	newParams := map[string]*Tensor{}
	for k, g := range grads {
		a.M[k] = Add(a.M[k], Mul(&Tensor{Val: 1.0 - a.Beta1}, Sub(g, a.M[k])))
		a.V[k] = Add(a.V[k], Mul(&Tensor{Val: 1.0 - a.Beta2}, Sub(g.Pow(2), a.V[k])))

		delta := Div(Mul(&Tensor{Val: lrT}, a.M[k]), Add(a.V[k].Sqrt(), &Tensor{Val: 1e-7}))
		newParams[k] = Sub(params[k], delta)
	}
	return newParams
}
