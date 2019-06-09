package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func MeanAll(t *Tensor) float64 {
	if len(t.Shape) == 2 {
		x := t.Mat
		return meanAllMat(x)
	}
	if len(t.Shape) == 3 {
		x := t.T3D
		return meanAllT3D(x)
	}
	if len(t.Shape) == 4 {
		x := t.T4D
		return meanAllT4D(x)
	}
	panic(1)
}

func meanAllMat(m *types.Matrix) float64 {
	return vec.Sum(m.Vector) / float64(len(m.Vector))
}

func meanAllT3D(m types.Tensor3D) float64 {
	return sumAllT3D(m) / float64(len(m))
}

func meanAllT4D(m types.Tensor4D) float64 {
	return sumAllT4D(m) / float64(len(m))
}

func Mean(t *Tensor, axis int) *Tensor {
	if len(t.Shape) == 2 {
		mat := t.Mat
		return &Tensor{
			Mat:   meanMat(mat, axis),
			Shape: t.Shape,
		}
	}
	panic(t)
}

func meanMat(m *types.Matrix, axis int) *types.Matrix {
	if axis == 0 {
		v := vec.Zeros(m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Sum(col) / float64(m.Rows)
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
			v[i] = vec.Sum(row) / float64(m.Columns)
		}
		return &types.Matrix{
			Vector:  v,
			Rows:    1,
			Columns: m.Rows,
		}
	}
	panic(m)
}
