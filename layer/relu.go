package layer

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type Relu struct {
	Mask []bool
}

func (r *Relu) forward(x *mat.Matrix) *mat.Matrix {
	v := x.Array
	r.Mask = make([]bool, len(v))
	out := make(vec.Vector, len(v))
	for i := 0; i < len(v); i++ {
		if v[i] <= 0 {
			r.Mask[i] = true
			out[i] = 0
		} else {
			out[i] = v[i]
		}
	}

	return &mat.Matrix{
		Array:   out,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func (r *Relu) backward(dout *mat.Matrix) *mat.Matrix {
	v := dout.Array
	for i := 0; i < len(v); i++ {
		if r.Mask[i] {
			v[i] = 0
		}
	}
	dx := &mat.Matrix{
		Array:   v,
		Rows:    dout.Rows,
		Columns: dout.Columns,
	}
	return dx
}
