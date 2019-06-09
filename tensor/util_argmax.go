package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func ArgMaxAll(t *Tensor) int {
	switch {
	case len(t.Shape) == 2:
		x := t.Mat
		return argMaxAllMat(x)
	case len(t.Shape) == 3:
		x := t.T3D
		return argMaxAllT3D(x)
	case len(t.Shape) == 4:
		x := t.T4D
		return argMaxAllT4D(x)
	}
	panic(1)
}

func argMaxAllMat(m *types.Matrix) int {
	return vec.ArgMax(m.Vector)
}

func argMaxAllT3D(m types.Tensor3D) int {
	max := 0
	for _, mat := range m {
		max += argMaxAllMat(mat)
	}
	return max
}
func argMaxAllT4D(m types.Tensor4D) int {
	max := 0
	for _, t3d := range m {
		max += argMaxAllT3D(t3d)
	}
	return max
}

func argMaxMat(m *types.Matrix, axis int) []int {
	if axis == 0 {
		v := make([]int, m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.ArgMax(col)
		}
		return v
	} else if axis == 1 {
		v := make([]int, m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.SliceRow(i)
			v[i] = vec.ArgMax(row)
		}
		return v
	}
	panic(m)
}

func ArgMax(t *Tensor, axis int) []int {
	if len(t.Shape) == 2 {
		return argMaxMat(t.Mat, axis)
	}
	panic(t)
}
