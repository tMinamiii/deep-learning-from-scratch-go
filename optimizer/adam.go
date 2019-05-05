package optimizer

import (
	"math"

	"github.com/naronA/zero_deeplearning/mat"
)

/*

class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

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
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
*/

type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Iter  int
	M     map[string]*mat.Matrix
	V     map[string]*mat.Matrix
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

func (a *Adam) Update(params, grads map[string]*mat.Matrix) map[string]*mat.Matrix {
	if a.M == nil {
		a.M = map[string]*mat.Matrix{}
		a.V = map[string]*mat.Matrix{}
		for k, v := range params {
			a.M[k] = mat.ZerosLike(v)
			a.V[k] = mat.ZerosLike(v)
		}
	}
	a.Iter++
	fIter := float64(a.Iter)
	// lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
	lrT := a.LR * math.Sqrt(1.0-math.Pow(a.Beta2, fIter)) / (1.0 - math.Pow(a.Beta1, fIter))

	newParams := map[string]*mat.Matrix{}
	for k, v := range params {
		// self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
		// self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
		a.M[k] = a.M[k].Add(v.Sub(a.M[k]).Mul(1.0 - a.Beta1))
		a.V[k] = a.V[k].Add(mat.Pow(v, 2).Sub(a.V[k]).Mul(1.0 - a.Beta2))

		// params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
		delta := (a.M[k].Mul(lrT)).Div(mat.Sqrt(a.V[k]).Add(1e-7))
		newParams[k] = params[k].Sub(delta)
	}
	return newParams
}
