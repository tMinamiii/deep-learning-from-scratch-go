package layer

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

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

func (a *Affine) Forward(x *mat.Matrix) *mat.Matrix {
	a.X = x
	out, err := mat.Dot(x, a.W).Add(a.B)
	if err != nil {
		panic(err)
	}
	return out
}

func (a *Affine) Backward(dout *mat.Matrix) *mat.Matrix {
	dx := mat.Dot(dout, a.W.T())
	a.DW = mat.Dot(a.X.T(), dout)
	a.DB = mat.Sum(dout, 0)
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
	dv := make(vec.Vector, len(v))
	for i := 0; i < len(v); i++ {
		if r.mask[i] {
			dv[i] = 0
		} else {
			dv[i] = v[i]
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

func (s *SoftmaxWithLoss) Forward(x, t *mat.Matrix) float64 {
	s.t = t
	s.y = mat.Softmax(x)
	s.loss = mat.CrossEntropyError(s.y, s.t)
	return s.loss
}

func (s *SoftmaxWithLoss) Backward(_ float64) *mat.Matrix {
	batchSize, _ := s.t.Shape()
	sub, err := s.y.Sub(s.t)
	if err != nil {
		panic(err)
	}

	dx, err := sub.Div(float64(batchSize))
	if err != nil {
		panic(err)
	}
	return dx
}
