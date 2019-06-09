package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func Sum(t *Tensor, axis int) *Tensor {
	if len(t.Shape) == 2 {
		mat := t.Mat
		return &Tensor{
			Mat:   sumMat(mat, axis),
			Shape: t.Shape,
		}
	}
	panic(t)
}

func sumMat(m *types.Matrix, axis int) *types.Matrix {
	if axis == 0 {
		v := vec.Zeros(m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Sum(col)
		}
		return &types.Matrix{
			Vector:  v,
			Rows:    1,
			Columns: m.Columns,
		}
	} else if axis == 1 {
		v := vec.Zeros(m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.SliceRow(i)
			v[i] = vec.Sum(row)
		}
		return &types.Matrix{
			Vector:  v,
			Rows:    1,
			Columns: m.Rows,
		}
	}
	panic(m)
}

func SumAll(t *Tensor) float64 {
	if len(t.Shape) == 2 {
		x := t.Mat
		return sumAllMat(x)
	}
	if len(t.Shape) == 3 {
		x := t.T3D
		return sumAllT3D(x)
	}
	if len(t.Shape) == 4 {
		x := t.T4D
		return sumAllT4D(x)
	}
	panic(1)
}

func sumAllMat(m *types.Matrix) float64 {
	return vec.Sum(m.Vector)
}

func sumAllT3D(m types.Tensor3D) float64 {
	sum := 0.0
	for _, mat := range m {
		sum += sumAllMat(mat)
	}
	return sum
}

func sumAllT4D(m types.Tensor4D) float64 {
	sum := 0.0
	for _, t3d := range m {
		sum += sumAllT3D(t3d)
	}
	return sum
}


