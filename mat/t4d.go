package mat

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/vec"
)

type Tensor4D []Tensor3D

func (t Tensor4D) Im2Col(fw, fh, stride, pad int) *Matrix {
	nv := vec.Vector{}
	for _, e := range t {
		nv = append(nv, e.Im2Col(fw, fh, stride, pad).Vector...)
	}
	N, C, H, _ := t.Shape()
	return &Matrix{
		Vector:  nv,
		Rows:    N * C * H,
		Columns: fw * fh * C,
	}
}

func (t Tensor4D) Size() int {
	n, c, h, w := t.Shape()
	return n * c * h * w
}

func (t Tensor4D) Element(n, c, h, w int) float64 {
	return t[n].Element(c, h, w)
}

func (t Tensor4D) Assign(value float64, n, c, h, w int) {
	t[n].Assign(value, c, h, w)
}

func (t Tensor4D) Flatten() vec.Vector {
	v := vec.Vector{}
	for _, e := range t {
		v = append(v, e.Flatten()...)
	}
	return v
}

func (t Tensor4D) Transpose(a, b, c, d int) Tensor4D {
	w, x, y, z := t.Shape()
	shape := []int{w, x, y, z}
	t4d := ZerosT4D(shape[a], shape[b], shape[c], shape[d])
	fmt.Println(t4d)
	fmt.Println(t)
	for i, e1t3d := range t {
		for j, e2mat := range e1t3d {
			for k := 0; k < e2mat.Rows; k++ {
				for l := 0; l < e2mat.Columns; l++ {
					oldIdx := []int{i, j, k, l}
					idx := make([]int, 4)
					idx[0] = oldIdx[a]
					idx[1] = oldIdx[b]
					idx[2] = oldIdx[c]
					idx[3] = oldIdx[d]
					// fmt.Println(i, j, k, l)
					// fmt.Println(" ", idx[0], idx[1], idx[2], idx[3])
					v := t.Element(i, j, k, l)
					t4d.Assign(v, idx[0], idx[1], idx[2], idx[3])
				}
			}
		}
	}
	return t4d
}

func (t Tensor4D) ReshapeToMat(row, col int) *Matrix {
	size := t.Size()
	if col == -1 {
		col = size / row
	} else if row == -1 {
		row = size / col
	}
	flat := t.Flatten()
	return &Matrix{
		Vector:  flat,
		Rows:    row,
		Columns: col,
	}
}

func (t Tensor4D) Window(x, y, h, w int) Tensor4D {
	newT4D := Tensor4D{}
	for _, mat := range t {
		newT4D = append(newT4D, mat.Window(x, y, h, w))
	}
	return newT4D
}

func (t Tensor4D) Pad(size int) Tensor4D {
	newT4D := Tensor4D{}
	for _, t3d := range t {
		padded := t3d.Pad(size)
		newT4D = append(newT4D, padded)
	}
	return newT4D
}

func (t Tensor4D) Shape() (int, int, int, int) {
	N := len(t)
	C := t[0].Channels()
	H, W := t[0][0].Shape()
	return N, C, H, W
}

func ZerosT4D(n, c, h, w int) Tensor4D {
	t4d := make(Tensor4D, n)
	for i := range t4d {
		t4d[i] = ZerosT3D(c, h, w)
	}
	return t4d
}

