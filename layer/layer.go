package layer

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type Affine struct {
	AW  *mat.Matrix
	Ab  *mat.Matrix
	Ax  *mat.Matrix
	AdW *mat.Matrix
	Adb *mat.Matrix
}

func NewAffine(w, b *mat.Matrix) *Affine {
	return &Affine{
		AW: w,
		Ab: b,
	}
}

func (a *Affine) Forward(x *mat.Matrix) *mat.Matrix {
	a.Ax = x
	out := mat.Dot(x, a.AW).Add(a.Ab.Vector)
	return out
}

func (a *Affine) Backward(dout *mat.Matrix) *mat.Matrix {
	dx := mat.Dot(dout, a.AW.T())
	a.AdW = mat.Dot(a.Ax.T(), dout)
	a.Adb = mat.Sum(dout, 0)
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

func (s *SoftmaxWithLoss) Forward(x, t *mat.Matrix) float64 {
	s.t = t
	s.y = mat.Softmax(x)
	s.loss = mat.CrossEntropyError(s.y, s.t)
	return s.loss
}

func (s *SoftmaxWithLoss) Backward(_ float64) *mat.Matrix {
	batchSize, _ := s.t.Shape()
	dx := (s.y.Sub(s.t)).Div(float64(batchSize))
	return dx
}

type Relu struct {
	mask []bool
}

func NewRelu() *Relu {
	return &Relu{}
}

func (r *Relu) Forward(x *mat.Matrix) *mat.Matrix {
	v := x.Vector
	r.mask = make([]bool, len(v))
	out := make(vec.Vector, len(v))
	for i := 0; i < len(v); i++ {
		if v[i] <= 0 {
			r.mask[i] = true
			out[i] = 0
		} else {
			out[i] = v[i]
		}
	}

	return &mat.Matrix{
		Vector:  out,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func (r *Relu) Backward(dout *mat.Matrix) *mat.Matrix {
	v := dout.Vector
	for i := 0; i < len(v); i++ {
		if r.mask[i] {
			v[i] = 0
		}
	}
	dx := &mat.Matrix{
		Vector:  v,
		Rows:    dout.Rows,
		Columns: dout.Columns,
	}
	return dx
}
