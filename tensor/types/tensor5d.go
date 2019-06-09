package types

import "github.com/naronA/zero_deeplearning/vec"

type Tensor5D []Tensor4D

func (t Tensor5D) Shape() (int, int, int, int, int) {
	a := len(t)
	b := len(t[0])
	c := len(t[0][0])
	d, e := t[0][0][0].Shape()
	return a, b, c, d, e

}

func (t Tensor5D) Element(b, n, c, h, w int) float64 {
	return t[b].Element(n, c, h, w)
}

func (t Tensor5D) Assign(value float64, b, n, c, h, w int) {
	t[b].Assign(value, n, c, h, w)
}

func (t Tensor5D) ReshapeTo2D(row, col int) *Matrix {
	a, b, c, d, e := t.Shape()
	size := a * b * c * d * e
	if row == -1 {
		row = size / col
	} else if col == -1 {
		col = size / row
	}

	return &Matrix{
		Vector:  t.Flatten(),
		Rows:    row,
		Columns: col,
	}
}

func (t Tensor5D) Flatten() vec.Vector {
	v := vec.Vector{}
	for _, e := range t {
		v = append(v, e.Flatten()...)
	}
	return v
}

func ZerosT5D(a, b, c, h, w int) Tensor5D {
	t5d := make(Tensor5D, a)
	for i := range t5d {
		t5d[i] = ZerosT4D(b, c, h, w)
	}
	return t5d
}

func ZerosLikeT5D(x Tensor5D) Tensor5D {
	t5d := make(Tensor5D, len(x))
	for i, v := range x {
		t5d[i] = ZerosLikeT4D(v)
	}
	return t5d
}
