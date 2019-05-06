package layer

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type Layer interface {
	Forward(*mat.Matrix, bool) *mat.Matrix
	Backward(*mat.Matrix) *mat.Matrix
}

type Affine struct {
	W  *mat.Matrix
	B  *mat.Matrix
	X  *mat.Matrix
	DW *mat.Matrix
	DB *mat.Matrix
}

func NewAffine(w, b *mat.Matrix) *Affine {
	return &Affine{
		W: w,
		B: b,
	}
}

func (af *Affine) Forward(x *mat.Matrix, _ bool) *mat.Matrix {
	af.X = x
	out := mat.Add(mat.Dot(x, af.W), af.B)
	return out
}

func (af *Affine) Backward(dout *mat.Matrix) *mat.Matrix {
	dx := mat.Dot(dout, af.W.T())
	af.DW = mat.Dot(af.X.T(), dout)
	af.DB = mat.Sum(dout, 0)
	return dx
}

type Sigmoid struct {
	Out *mat.Matrix
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (si *Sigmoid) Forward(x *mat.Matrix, _ bool) *mat.Matrix {
	minusX := mat.Mul(x, -1.0)
	exp := mat.Exp(minusX)
	plusX := mat.Add(exp, 1.0)
	out := mat.Pow(plusX, -1)
	si.Out = out
	return out
}

func (si *Sigmoid) Backward(dout *mat.Matrix) *mat.Matrix {
	minus := mat.Mul(si.Out, -1.0)
	sub := mat.Add(minus, 1.0)
	mul := mat.Mul(dout, sub)
	dx := mat.Mul(mul, si.Out)
	return dx
}

type ReLU struct {
	mask []bool
}

func NewRelu() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *mat.Matrix, _ bool) *mat.Matrix {
	v := x.Vector
	r.mask = make([]bool, len(v))
	out := vec.ZerosLike(v)
	for i, e := range v {
		if e <= 0 {
			r.mask[i] = true
			out[i] = 0
		} else {
			out[i] = e
		}
	}

	return &mat.Matrix{
		Vector:  out,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func (r *ReLU) Backward(dout *mat.Matrix) *mat.Matrix {
	v := dout.Vector
	dv := vec.ZerosLike(v)
	for i, e := range v {
		if r.mask[i] {
			dv[i] = 0
		} else {
			dv[i] = e
		}
	}
	dx := &mat.Matrix{
		Vector:  dv,
		Rows:    dout.Rows,
		Columns: dout.Columns,
	}
	return dx
}

type SoftmaxWithLoss struct {
	loss float64
	y    *mat.Matrix
	t    *mat.Matrix
}

func NewSfotmaxWithLoss() *SoftmaxWithLoss {
	return &SoftmaxWithLoss{}
}

func (so *SoftmaxWithLoss) Forward(x, t *mat.Matrix) float64 {
	so.t = t
	so.y = mat.Softmax(x)
	so.loss = mat.CrossEntropyError(so.y, so.t)
	return so.loss
}

func (so *SoftmaxWithLoss) Backward(_ float64) *mat.Matrix {
	batchSize, _ := so.t.Shape()
	sub := mat.Sub(so.y, so.t)
	dx := mat.Div(sub, batchSize)
	return dx
}
