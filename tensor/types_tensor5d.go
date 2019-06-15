package tensor

import "github.com/naronA/zero_deeplearning/vec"

type Tensor5D []Tensor4D

func (t Tensor5D) Shape() (int, int, int, int, int) {
	a := len(t)
	b := len(t[0])
	c := len(t[0][0])
	d, e := t[0][0][0].Shape()
	return a, b, c, d, e

}

func (t Tensor5D) element(b, n, c, h, w int) float64 {
	return t[b].element(n, c, h, w)
}

func (t Tensor5D) assign(value float64, b, n, c, h, w int) {
	t[b].assign(value, n, c, h, w)
}

func (t Tensor5D) flatten() vec.Vector {
	v := vec.Vector{}
	for _, e := range t {
		v = append(v, e.flatten()...)
	}
	return v
}

func zerosT5D(a, b, c, h, w int) Tensor5D {
	t5d := make(Tensor5D, a)
	for i := range t5d {
		t5d[i] = zerosT4D(b, c, h, w)
	}
	return t5d
}

func zerosLikeT5D(x Tensor5D) Tensor5D {
	t5d := make(Tensor5D, len(x))
	for i, v := range x {
		t5d[i] = zerosLikeT4D(v)
	}
	return t5d
}

func (t Tensor5D) window(x, y, h, w int) Tensor5D {
	newT5D := make(Tensor5D, len(t))
	for i, t4d := range t {
		newT5D[i] = t4d.window(x, y, h, w)
	}
	return newT5D
}

func (t Tensor5D) transpose(a, b, c, d, e int) Tensor5D {
	u, v, w, x, y := t.Shape()
	shape := []int{u, v, w, x, y}
	t5d := zerosT5D(shape[a], shape[b], shape[c], shape[d], shape[e])
	for i, et4d := range t {
		for j, et3d := range et4d {
			for k, emat := range et3d {
				for l := 0; l < emat.Rows; l++ {
					for n := 0; n < emat.Columns; n++ {
						oldIdx := []int{i, j, k, l, n}
						idx := make([]int, 5)
						idx[0] = oldIdx[a]
						idx[1] = oldIdx[b]
						idx[2] = oldIdx[c]
						idx[3] = oldIdx[d]
						idx[4] = oldIdx[e]
						// fmt.Println(i, j, k, l)
						// fmt.Println(" ", idx[0], idx[1], idx[2], idx[3])
						v := t.element(i, j, k, l, n)
						t5d.assign(v, idx[0], idx[1], idx[2], idx[3], idx[4])
					}
				}
			}
		}
	}
	return t5d
}

func (t Tensor5D) pad(size int) Tensor5D {
	newT5D := make(Tensor5D, len(t))
	for i, t4d := range t {
		padded := t4d.pad(size)
		newT5D[i] = padded
	}
	return newT5D
}

func (t Tensor5D) equal(x Tensor5D) bool {
	for i := range t {
		if !t[i].equal(x[i]) {
			return false
		}
	}
	return true
}

func (t Tensor5D) reshapeToMat(row, col int) *Matrix {
	a, b, c, d, e := t.Shape()
	size := a * b * c * d * e
	if row == -1 {
		row = size / col
	} else if col == -1 {
		col = size / row
	}

	return &Matrix{
		Vector:  t.flatten(),
		Rows:    row,
		Columns: col,
	}
}
