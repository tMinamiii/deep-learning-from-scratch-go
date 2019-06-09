package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func maxMat(m *types.Matrix, axis int) vec.Vector {
	if axis == 0 {
		v := vec.Zeros(m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Max(col)
		}
		return v
	} else if axis == 1 {
		v := vec.Zeros(m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.SliceRow(i)
			v[i] = vec.Max(row)
		}
		return v
	}
	panic(m)
}

func Max(t *Tensor, axis int) *Tensor {
	if len(t.Shape) == 2 {
		vec := maxMat(t.Mat, axis)
		return &Tensor{
			Vec:   vec,
			Shape: []int{len(vec)},
		}
	}
	panic(t)
}

func maxAllMat(m *types.Matrix) float64 {
	return vec.Max(m.Vector)
}

func maxAllT3D(m types.Tensor3D) float64 {
	max := 0.0
	for _, mat := range m {
		max += maxAllMat(mat)
	}
	return max
}
func maxAllT4D(m types.Tensor4D) float64 {
	max := 0.0
	for _, t3d := range m {
		max += maxAllT3D(t3d)
	}
	return max
}

func MaxAll(t *Tensor) float64 {
	if len(t.Shape) == 2 {
		x := t.Mat
		return maxAllMat(x)
	}
	if len(t.Shape) == 3 {
		x := t.T3D
		return maxAllT3D(x)
	}
	if len(t.Shape) == 4 {
		x := t.T4D
		return maxAllT4D(x)
	}
	panic(1)
}
