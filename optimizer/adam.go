package optimizer

import (
	"math"

	"github.com/naronA/zero_deeplearning/mat"
)

/*
class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
*/

type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Iter  int
	M     map[string]*mat.Matrix
	V     map[string]*mat.Matrix
}

func NewAdam(lr, beta1, beta2 float64) *Adam {
	return &Adam{
		LR:    lr,
		Beta1: beta1,
		Beta2: beta2,
		Iter:  0,
		M:     nil,
		V:     nil,
	}
}

func (self *Adam) Update(params, grads map[string]*mat.Matrix) {
	if self.M == nil {
		self.M = map[string]*mat.Matrix{}
		self.V = map[string]*mat.Matrix{}
	}
	self.Iter++
	fIter := float64(self.Iter)
	lrT := self.LR * math.Sqrt(1.0-math.Pow(self.Beta2, fIter)) / (1.0 - math.Pow(self.Beta1, fIter))

	newParams := map[string]*mat.Matrix{}
	for k, v := range params {

		// self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
		self.M[k] = self.M[k].Add(v.Sub(self.M[k]).Mul(1.0 - self.Beta1))

		// self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
		self.V[k] = self.V[k].Add(mat.Pow(v, 2).Sub(self.V[k]).Mul(1.0 - self.Beta2))

		// params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
		delta := self.M[k].Mul(lrT).Div(mat.Sqrt(self.V[k]).Add(1e-10))
		newParams[k] = params[k].Sub(delta)
	}
}
