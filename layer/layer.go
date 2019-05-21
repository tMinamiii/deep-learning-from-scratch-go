package layer

import (
	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

type Layer interface {
	Forward(*num.Matrix, bool) *num.Matrix
	Backward(*num.Matrix) *num.Matrix
}

type Affine struct {
	W  *num.Matrix
	B  *num.Matrix
	X  *num.Matrix
	DW *num.Matrix
	DB *num.Matrix
}

func NewAffine(w, b *num.Matrix) *Affine {
	return &Affine{
		W: w,
		B: b,
	}
}

func (af *Affine) Forward(x *num.Matrix, _ bool) *num.Matrix {
	af.X = x
	out := num.Add(num.Dot(x, af.W), af.B)
	return out
}

func (af *Affine) Backward(dout *num.Matrix) *num.Matrix {
	dx := num.Dot(dout, af.W.T())
	af.DW = num.Dot(af.X.T(), dout)
	af.DB = num.Sum(dout, 0)
	return dx
}

type Sigmoid struct {
	Out *num.Matrix
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (si *Sigmoid) Forward(x *num.Matrix, _ bool) *num.Matrix {
	minusX := num.Mul(x, -1.0)
	exp := num.Exp(minusX)
	plusX := num.Add(exp, 1.0)
	out := num.Pow(plusX, -1)
	si.Out = out
	return out
}

func (si *Sigmoid) Backward(dout *num.Matrix) *num.Matrix {
	minus := num.Mul(si.Out, -1.0)
	sub := num.Add(minus, 1.0)
	mul := num.Mul(dout, sub)
	dx := num.Mul(mul, si.Out)
	return dx
}

type ReLU struct {
	mask []bool
}

func NewRelu() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *num.Matrix, _ bool) *num.Matrix {
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

	return &num.Matrix{
		Vector:  out,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func (r *ReLU) Backward(dout *num.Matrix) *num.Matrix {
	v := dout.Vector
	dv := vec.ZerosLike(v)
	for i, e := range v {
		if r.mask[i] {
			dv[i] = 0
		} else {
			dv[i] = e
		}
	}
	dx := &num.Matrix{
		Vector:  dv,
		Rows:    dout.Rows,
		Columns: dout.Columns,
	}
	return dx
}

type SoftmaxWithLoss struct {
	loss float64
	y    *num.Matrix
	t    *num.Matrix
}

func NewSfotmaxWithLoss() *SoftmaxWithLoss {
	return &SoftmaxWithLoss{}
}

func (so *SoftmaxWithLoss) Forward(x, t *num.Matrix) float64 {
	so.t = t
	so.y = num.Softmax(x)
	so.loss = num.CrossEntropyError(so.y, so.t)
	return so.loss
}

func (so *SoftmaxWithLoss) Backward(_ float64) *num.Matrix {
	batchSize, _ := so.t.Shape()
	sub := num.Sub(so.y, so.t)
	dx := num.Div(sub, batchSize)
	return dx
}
