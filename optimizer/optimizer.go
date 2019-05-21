package optimizer

import "github.com/naronA/zero_deeplearning/num"

type Optimizer interface {
	Update(map[string]*num.Matrix, map[string]*num.Matrix) map[string]*num.Matrix
}

type SGD struct {
	LR float64
}

func NewSGD(lr float64) *SGD {
	return &SGD{
		LR: lr,
	}
}

func (sgd *SGD) Update(params, grads map[string]*num.Matrix) map[string]*num.Matrix {
	newParams := map[string]*num.Matrix{}
	for k, g := range grads {
		newParams[k] = num.Sub(params[k], num.Mul(g, sgd.LR))
	}
	return newParams
}

type Momentum struct {
	LR       float64
	Momentum float64
	V        map[string]*num.Matrix
}

func NewMomentum(lr float64) *Momentum {
	return &Momentum{
		LR:       lr,
		Momentum: 0.9,
		V:        nil,
	}
}

func (mo *Momentum) Update(params, grads map[string]*num.Matrix) map[string]*num.Matrix {
	if mo.V == nil {
		mo.V = map[string]*num.Matrix{}
		for k, v := range params {
			mo.V[k] = num.ZerosLike(v)
		}
	}

	newParams := map[string]*num.Matrix{}
	for k, g := range grads {
		mo.V[k] = num.Sub(num.Mul(mo.Momentum, mo.V[k]), num.Mul(mo.LR, g))
		newParams[k] = num.Add(params[k], mo.V[k])
	}
	return newParams
}

type AdaGrad struct {
	LR float64
	H  map[string]*num.Matrix
}

func NewAdaGrad(lr float64) *AdaGrad {
	return &AdaGrad{
		LR: lr,
		H:  nil,
	}
}

func (ad *AdaGrad) Update(params, grads map[string]*num.Matrix) map[string]*num.Matrix {
	if ad.H == nil {
		ad.H = map[string]*num.Matrix{}
		for k, v := range params {
			ad.H[k] = num.ZerosLike(v)
		}
	}
	newParams := map[string]*num.Matrix{}
	for k, g := range grads {
		ad.H[k] = num.Add(ad.H[k], num.Pow(g, 2))
		delta := num.Div(num.Mul(ad.LR, g), num.Add(num.Sqrt(ad.H[k]), 1e-7))
		newParams[k] = num.Sub(params[k], delta)
	}
	return newParams
}
