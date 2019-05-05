package layer

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type Layer interface {
	Forward(*mat.Matrix) *mat.Matrix
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

func (self *Affine) Forward(x *mat.Matrix) *mat.Matrix {
	self.X = x
	out := mat.Dot(x, self.W).Add(self.B)
	return out
}

func (self *Affine) Backward(dout *mat.Matrix) *mat.Matrix {
	dx := mat.Dot(dout, self.W.T())
	self.DW = mat.Dot(self.X.T(), dout)
	self.DB = mat.Sum(dout, 0)
	return dx
}

type Sigmoid struct {
	Out *mat.Matrix
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (self *Sigmoid) Forward(x *mat.Matrix) *mat.Matrix {
	minusX := x.Mul(-1.0)
	exp := mat.Exp(minusX)
	plusX := exp.Add(1.0)
	out := mat.Pow(plusX, -1)
	self.Out = out
	return out
}

func (self *Sigmoid) Backward(dout *mat.Matrix) *mat.Matrix {
	minus := self.Out.Mul(-1.0)
	sub := minus.Add(1.0)
	mul := dout.Mul(sub)
	dx := mul.Mul(self.Out)
	return dx
}

type ReLU struct {
	mask []bool
}

func NewRelu() *ReLU {
	return &ReLU{}
}

func (self *ReLU) Forward(x *mat.Matrix) *mat.Matrix {
	v := x.Vector
	self.mask = make([]bool, len(v))
	out := vec.ZerosLike(v)
	for i, e := range v {
		if e <= 0 {
			self.mask[i] = true
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

func (self *ReLU) Backward(dout *mat.Matrix) *mat.Matrix {
	v := dout.Vector
	dv := vec.ZerosLike(v)
	for i, e := range v {
		if self.mask[i] {
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

func (self *SoftmaxWithLoss) Forward(x, t *mat.Matrix) float64 {
	self.t = t
	self.y = mat.Softmax(x)
	self.loss = mat.CrossEntropyError(self.y, self.t)
	return self.loss
}

func (self *SoftmaxWithLoss) Backward(_ float64) *mat.Matrix {
	batchSize, _ := self.t.Shape()
	sub := self.y.Sub(self.t)
	dx := sub.Div(float64(batchSize))
	return dx
}
