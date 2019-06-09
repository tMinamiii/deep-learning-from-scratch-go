package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func Pow(m *Tensor, p float64) *Tensor {
	if len(m.Shape) == 2 {
		mat := m.Mat
		return &Tensor{
			Mat:   powMat(mat, p),
			Shape: m.Shape,
		}
	}
	if len(m.Shape) == 3 {
		t3d := m.T3D
		return &Tensor{
			T3D:   powT3D(t3d, p),
			Shape: m.Shape,
		}
	}
	if len(m.Shape) == 4 {
		t4d := m.T4D
		return &Tensor{
			T4D:   powT4D(t4d, p),
			Shape: m.Shape,
		}
	}
	return nil
}

func powMat(m *types.Matrix, p float64) *types.Matrix {
	mat := vec.Pow(m.Vector, p)
	return &types.Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func powT3D(m types.Tensor3D, p float64) types.Tensor3D {
	t3d := make([]*types.Matrix, len(m))
	for i, mat := range t3d {
		t3d[i] = powMat(mat, p)
	}
	return t3d
}

func powT4D(m types.Tensor4D, p float64) types.Tensor4D {
	t4d := make([]types.Tensor3D, len(m))
	for i, t3d := range t4d {
		t4d[i] = powT3D(t3d, p)
	}
	return t4d
}
