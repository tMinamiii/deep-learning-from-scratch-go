package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func (t *Tensor) Pad(pad int) *Tensor {
	switch len(t.Shape) {
	case 2:
		return &Tensor{Mat: padMat(t.Mat, pad), Shape: t.Shape}
	case 3:
		return &Tensor{T3D: padT3D(t.T3D, pad), Shape: t.Shape}
	case 4:
		return &Tensor{T4D: padT4D(t.T4D, pad), Shape: t.Shape}
	case 5:
		return &Tensor{T5D: padT5D(t.T5D, pad), Shape: t.Shape}
	case 6:
		return &Tensor{T6D: padT6D(t.T6D, pad), Shape: t.Shape}
	}
	panic(t)
}

func padMat(m *types.Matrix, pad int) *types.Matrix {
	if pad == 0 {
		return &types.Matrix{
			Vector:  m.Vector,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	}
	col := m.Columns
	newVec := make(vec.Vector, 0, m.Rows+2*pad)
	rowPad := vec.Zeros(col + 2*pad)
	for j := 0; j < pad; j++ {
		newVec = append(newVec, rowPad...)
	}
	for j := 0; j < m.Rows; j++ {
		srow := m.SliceRow(j)
		for k := 0; k < pad; k++ {
			newVec = append(newVec, 0)
		}
		newVec = append(newVec, srow...)
		for k := 0; k < pad; k++ {
			newVec = append(newVec, 0)
		}
	}
	for j := 0; j < pad; j++ {
		newVec = append(newVec, rowPad...)
	}
	return &types.Matrix{
		Vector:  newVec,
		Rows:    m.Rows + 2*pad,
		Columns: m.Columns + 2*pad,
	}

}

func padT3D(t types.Tensor3D, size int) types.Tensor3D {
	newT3D := make(types.Tensor3D, len(t))
	for i, m := range t {
		newT3D[i] = padMat(m, size)
	}
	return newT3D
}

func padT4D(t types.Tensor4D, size int) types.Tensor4D {
	newT4D := make(types.Tensor4D, len(t))
	for i, t3d := range t {
		padded := padT3D(t3d, size)
		newT4D[i] = padded
	}
	return newT4D
}

func padT5D(t types.Tensor5D, size int) types.Tensor5D {
	newT5D := make(types.Tensor5D, len(t))
	for i, t4d := range t {
		padded := padT4D(t4d, size)
		newT5D[i] = padded
	}
	return newT5D
}

func padT6D(t types.Tensor6D, size int) types.Tensor6D {
	newT6D := make(types.Tensor6D, len(t))
	for i, t5d := range t {
		padded := padT5D(t5d, size)
		newT6D[i] = padded
	}
	return newT6D
}
