package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

/* Dependent */
func (t *Tensor) Transpose(axis []int) *Tensor {
	switch {
	case len(t.Shape) == 2:
		mat := t.Mat
		trans := transposeMat(mat, axis[0], axis[1])
		r, c := trans.Shape()
		shape := []int{r, c}
		return &Tensor{
			Mat:   trans,
			Shape: shape,
		}
	case len(t.Shape) == 3:
		trans := transposeT3D(t.T3D, axis[0], axis[1], axis[2])
		c, h, w := trans.Shape()
		return &Tensor{
			T3D:   trans,
			Shape: []int{c, h, w},
		}
	case len(t.Shape) == 4:
		trans := transposeT4D(t.T4D, axis[0], axis[1], axis[2], axis[3])
		n, c, h, w := trans.Shape()
		return &Tensor{
			T4D:   trans,
			Shape: []int{n, c, h, w},
		}
	case len(t.Shape) == 5:
		trans := transposeT5D(t.T5D, axis[0], axis[1], axis[2], axis[3], axis[4])
		a, n, c, h, w := trans.Shape()
		return &Tensor{
			T5D:   trans,
			Shape: []int{a, n, c, h, w},
		}
	case len(t.Shape) == 6:
		trans := transposeT6D(t.T6D, axis[0], axis[1], axis[2], axis[3], axis[4], axis[5])
		a, b, c, d, e, f := trans.Shape()
		return &Tensor{
			T6D:   trans,
			Shape: []int{a, b, c, d, e, f},
		}
	}
	panic(t)
}

func transposeMat(m *types.Matrix, a, b int) *types.Matrix {
	trans := make(vec.Vector, m.Rows*m.Columns)
	if a == 0 && b == 1 {
		for i := 0; i < m.Rows; i++ {
			col := m.SliceRow(i)
			for j := 0; j < len(col); j++ {
				trans[i*len(col)+j] = col[j]
			}
		}
		return &types.Matrix{
			Vector:  trans,
			Rows:    m.Rows,
			Columns: m.Columns,
		}

	}
	for i := 0; i < m.Columns; i++ {
		col := m.SliceColumn(i)
		// trans = append(trans, col...)
		for j := 0; j < len(col); j++ {
			trans[i*len(col)+j] = col[j]
		}
	}
	return &types.Matrix{
		Vector:  trans,
		Rows:    m.Columns,
		Columns: m.Rows,
	}
}
func transposeT3D(t types.Tensor3D, a, b, c int) types.Tensor3D {
	x, y, z := t.Shape()
	shape := []int{x, y, z}
	t3d := types.ZerosT3D(shape[a], shape[b], shape[c])
	for i, mat := range t {
		for j := 0; j < mat.Rows; j++ {
			for k := 0; k < mat.Columns; k++ {
				oldIdx := []int{i}
				idx := make([]int, 3)
				idx[0] = oldIdx[a]
				idx[1] = oldIdx[b]
				idx[2] = oldIdx[c]
				v := t.Element(i, j, k)
				t3d.Assign(v, idx[0], idx[1], idx[2])
			}
		}
	}
	return t3d
}
func transposeT4D(t types.Tensor4D, a, b, c, d int) types.Tensor4D {
	w, x, y, z := t.Shape()
	shape := []int{w, x, y, z}
	t4d := types.ZerosT4D(shape[a], shape[b], shape[c], shape[d])
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
					v := t.Element(i, j, k, l)
					t4d.Assign(v, idx[0], idx[1], idx[2], idx[3])
				}
			}
		}
	}
	return t4d
}

func transposeT5D(t types.Tensor5D, a, b, c, d, e int) types.Tensor5D {
	u, v, w, x, y := t.Shape()
	shape := []int{u, v, w, x, y}
	t5d := types.ZerosT5D(shape[a], shape[b], shape[c], shape[d], shape[e])
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
						v := t.Element(i, j, k, l, n)
						t5d.Assign(v, idx[0], idx[1], idx[2], idx[3], idx[4])
					}
				}
			}
		}
	}
	return t5d
}

func transposeT6D(t types.Tensor6D, a, b, c, d, e, f int) types.Tensor6D {
	u, v, w, x, y, z := t.Shape()
	shape := []int{u, v, w, x, y, z}
	t6d := types.ZerosT6D(shape[a], shape[b], shape[c], shape[d], shape[e], shape[f])
	for i, et5d := range t {
		for j, et4d := range et5d {
			for k, et3d := range et4d {
				for l, emat := range et3d {
					for n := 0; n < emat.Rows; n++ {
						for m := 0; m < emat.Columns; m++ {
							oldIdx := []int{i, j, k, l, n, m}
							idx := make([]int, 6)
							idx[0] = oldIdx[a]
							idx[1] = oldIdx[b]
							idx[2] = oldIdx[c]
							idx[3] = oldIdx[d]
							idx[4] = oldIdx[e]
							idx[5] = oldIdx[f]
							// fmt.Println(i, j, k, l)
							// fmt.Println(" ", idx[0], idx[1], idx[2], idx[3])
							v := t.Element(i, j, k, l, n, m)
							t6d.Assign(v, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5])
						}
					}
				}
			}
		}
	}
	return t6d
}
