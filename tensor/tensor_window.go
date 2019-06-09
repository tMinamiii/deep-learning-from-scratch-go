package tensor

import "github.com/naronA/zero_deeplearning/tensor/types"

func (t *Tensor) Window(x, y, h, w int) *Tensor {
	if len(t.Shape) == 2 {
		m := t.Mat
		return &Tensor{
			Mat:   windowMat(m, x, y, h, w),
			Shape: t.Shape,
		}
	}
	panic(t)
}

func windowMat(m *types.Matrix, x, y, h, w int) *types.Matrix {
	mat := zerosMat([]int{h, w})
	for i := x; i < x+h; i++ {
		for j := y; j < y+w; j++ {
			mat.Vector[(i-x)*w+(j-y)] = m.Element(i, j)
		}
	}
	return mat
}

func windowT3D(t types.Tensor3D, x, y, h, w int) types.Tensor3D {
	newT3D := make(types.Tensor3D, len(t))
	for i, mat := range t {
		newT3D[i] = windowMat(mat, x, y, h, w)
	}
	return newT3D
}

func windowT4D(t types.Tensor4D, x, y, h, w int) types.Tensor4D {
	newT4D := make(types.Tensor4D, len(t))
	for i, mat := range t {
		newT4D[i] = windowT3D(mat, x, y, h, w)
	}
	return newT4D
}

func windowT5D(t types.Tensor5D, x, y, h, w int) types.Tensor5D {
	newT5D := make(types.Tensor5D, len(t))
	for i, mat := range t {
		newT5D[i] = windowT4D(mat, x, y, h, w)
	}
	return newT5D
}

func windowT6D(t types.Tensor6D, x, y, h, w int) types.Tensor6D {
	newT6D := make(types.Tensor6D, len(t))
	for i, mat := range t {
		newT6D[i] = windowT5D(mat, x, y, h, w)
	}
	return newT6D
}
