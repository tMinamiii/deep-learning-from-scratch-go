package optimizer

import (
	"math"

	"github.com/naronA/zero_deeplearning/num"
)

type AnyOptimizer interface {
	Update(map[string]interface{}, map[string]interface{}) map[string]interface{}
}

type AdamAny struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Iter  int
	M     map[string]interface{}
	V     map[string]interface{}
}

func NewAdamAny(lr float64) *AdamAny {
	return &AdamAny{
		LR:    lr,
		Beta1: 0.9,
		Beta2: 0.999,
		Iter:  0,
		M:     nil,
		V:     nil,
	}
}

func (a *AdamAny) Update(params, grads map[string]interface{}) map[string]interface{} {
	if a.M == nil {
		a.M = map[string]interface{}{}
		a.V = map[string]interface{}{}
		for k, v := range params {
			if t4d, ok := v.(num.Tensor4D); ok {
				a.M[k] = num.ZerosLikeT4D(t4d)
				a.V[k] = num.ZerosLikeT4D(t4d)
			}
			if mat, ok := v.(*num.Matrix); ok {
				a.M[k] = num.ZerosLike(mat)
				a.V[k] = num.ZerosLike(mat)
			}

		}
	}
	a.Iter++
	fIter := float64(a.Iter)
	// lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
	lrT := a.LR * math.Sqrt(1.0-math.Pow(a.Beta2, fIter)) / (1.0 - math.Pow(a.Beta1, fIter))

	newParams := map[string]interface{}{}
	for k, g := range grads {
		if t4d, ok := g.(num.Tensor4D); ok {
			a.M[k] = num.AddT4D(a.M[k], num.MulT4D(1.0-a.Beta1, num.SubT4D(g, a.M[k])))
			a.V[k] = num.AddT4D(a.V[k], num.MulT4D(1.0-a.Beta2, num.SubT4D(num.PowT4D(t4d, 2), a.V[k])))

			delta := num.DivT4D(num.MulT4D(lrT, a.M[k]), num.AddT4D(num.SqrtT4D(a.V[k].(num.Tensor4D)), 1e-7))
			newParams[k] = num.SubT4D(params[k], delta)
		}
		if mat, ok := g.(*num.Matrix); ok {
			a.M[k] = num.Add(a.M[k], num.Mul(1.0-a.Beta1, num.Sub(g, a.M[k])))
			a.V[k] = num.Add(a.V[k], num.Mul(1.0-a.Beta2, num.Sub(num.Pow(mat, 2), a.V[k])))

			delta := num.Div(num.Mul(lrT, a.M[k]), num.Add(num.Sqrt(a.V[k].(*num.Matrix)), 1e-7))
			newParams[k] = num.Sub(params[k], delta)
		}
	}
	return newParams
}
