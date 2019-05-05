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

func (self *SGD) Update(params, grads map[string]*mat.Matrix) map[string]*mat.Matrix {
	newParams := map[string]*mat.Matrix{}
	for k, v := range grads {
		newParams[k] = params[k].Sub(v.Mul(self.LR))
	}
	return newParams
}

type Momentum struct {
	LR       float64
	Momentum float64
	V        map[string]*mat.Matrix
}

func NewMomentum(lr, momentum float64) *Momentum {
	return &Momentum{
		LR:       lr,
		Momentum: momentum,
		V:        nil,
	}
}

func (self *Momentum) Update(params, grads map[string]*mat.Matrix) map[string]*mat.Matrix {
	if self.V == nil {
		self.V = map[string]*mat.Matrix{}
		for k, v := range params {
			self.V[k] = mat.ZerosLike(v)
		}
	}

	newParams := map[string]*mat.Matrix{}
	for k, v := range grads {
		self.V[k] = self.V[k].Mul(self.Momentum).Sub(v.Mul(self.LR))
		newParams[k] = params[k].Add(self.V[k])
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

func (self *AdaGrad) Update(params, grads map[string]*mat.Matrix) map[string]*mat.Matrix {
	if self.H == nil {
		self.H = map[string]*mat.Matrix{}
		for k, v := range params {
			self.H[k] = mat.ZerosLike(v)
		}
	}
	newParams := map[string]*mat.Matrix{}
	for k, v := range grads {
		self.H[k] = mat.Pow(v, 2).Add(self.H[k])
		delta := v.Mul(self.LR).Div(mat.Sqrt(self.H[k]).Add(1e-10))
		newParams[k] = params[k].Sub(delta)
	}
	return newParams
}
