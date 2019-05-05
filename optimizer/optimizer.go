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
		newParams[k] = mat.Sub(params[k], mat.Mul(v, sgd.LR))
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
		mo.V[k] = mat.Sub(mat.Mul(mo.V[k], mo.Momentum), mat.Mul(mo.LR, v))
		newParams[k] = mat.Add(params[k], mo.V[k])
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
		ad.H[k] = mat.Add(mat.Pow(v, 2), ad.H[k])
		delta := mat.Div(mat.Mul(v, ad.LR), mat.Add(mat.Sqrt(ad.H[k]), 1e-7))
		newParams[k] = mat.Sub(params[k], delta)
	}
	return newParams
}
