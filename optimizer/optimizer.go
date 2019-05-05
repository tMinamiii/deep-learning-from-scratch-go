package optimizer

import "github.com/naronA/zero_deeplearning/mat"

type Optimizer interface {
	Update(map[string]*mat.Matrix, map[string]*mat.Matrix) map[string]*mat.Matrix
}

type SGD struct {
	LR float64
}

func NewSGD(lr float64) *SGD {
	return &SGD{
		LR: lr,
	}
}

func (sgd *SGD) Update(params, grads map[string]*mat.Matrix) map[string]*mat.Matrix {
	newParams := map[string]*mat.Matrix{}
	for k, v := range grads {
		newParams[k] = params[k].Sub(v.Mul(sgd.LR))
	}
	return newParams
}

type Momentum struct {
	LR       float64
	Momentum float64
	V        map[string]*mat.Matrix
}

func NewMomentum(lr float64) *Momentum {
	return &Momentum{
		LR:       lr,
		Momentum: 0.9,
		V:        nil,
	}
}

func (mo *Momentum) Update(params, grads map[string]*mat.Matrix) map[string]*mat.Matrix {
	if mo.V == nil {
		mo.V = map[string]*mat.Matrix{}
		for k, v := range params {
			mo.V[k] = mat.ZerosLike(v)
		}
	}

	newParams := map[string]*mat.Matrix{}
	for k, v := range grads {
		mo.V[k] = mo.V[k].Mul(mo.Momentum).Sub(v.Mul(mo.LR))
		newParams[k] = params[k].Add(mo.V[k])
	}
	return newParams
}

type AdaGrad struct {
	LR float64
	H  map[string]*mat.Matrix
}

func NewAdaGrad(lr float64) *AdaGrad {
	return &AdaGrad{
		LR: lr,
		H:  nil,
	}
}

func (ad *AdaGrad) Update(params, grads map[string]*mat.Matrix) map[string]*mat.Matrix {
	if ad.H == nil {
		ad.H = map[string]*mat.Matrix{}
		for k, v := range params {
			ad.H[k] = mat.ZerosLike(v)
		}
	}
	newParams := map[string]*mat.Matrix{}
	for k, v := range grads {
		ad.H[k] = mat.Pow(v, 2).Add(ad.H[k])
		delta := v.Mul(ad.LR).Div(mat.Sqrt(ad.H[k]).Add(1e-7))
		newParams[k] = params[k].Sub(delta)
	}
	return newParams
}
