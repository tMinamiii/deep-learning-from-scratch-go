package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func Sqrt(m *Tensor) *Tensor {
	if len(m.Shape) == 2 {
		mat := m.Mat
		return &Tensor{
			Mat:   sqrtMat(mat),
			Shape: m.Shape,
		}
	}
	if len(m.Shape) == 3 {
		t3d := m.T3D
		return &Tensor{
			T3D:   sqrtT3D(t3d),
			Shape: m.Shape,
		}
	}
	if len(m.Shape) == 4 {
		t4d := m.T4D
		return &Tensor{
			T4D:   sqrtT4D(t4d),
			Shape: m.Shape,
		}
	}
	return nil
}

func sqrtMat(m *types.Matrix) *types.Matrix {
	mat := vec.Sqrt(m.Vector)
	return &types.Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func sqrtT3D(m types.Tensor3D) types.Tensor3D {
	t3d := make([]*types.Matrix, len(m))
	for i, mat := range t3d {
		t3d[i] = sqrtMat(mat)
	}
	return t3d
}

func sqrtT4D(m types.Tensor4D) types.Tensor4D {
	t4d := make([]types.Tensor3D, len(m))
	for i, t3d := range t4d {
		t4d[i] = sqrtT3D(t3d)
	}
	return t4d
}
