package layer

import "github.com/naronA/zero_deeplearning/mat"

type Affine struct {
	W  *mat.Matrix
	b  *mat.Matrix
	x  *mat.Matrix
	dW *mat.Matrix
	db *mat.Matrix
}

func (a *Affine) forward(x *mat.Matrix) *mat.Matrix {
	a.x = x
	out := mat.Dot(x, a.W).Add(a.b)
	return out
}

func (a *Affine) backward(dout *mat.Matrix) *mat.Matrix {
	dx := mat.Dot(dout, a.W)
	a.dW = mat.Dot(a.x, dout)
	a.db = mat.Sum(dout, 0)
	return dx
}
