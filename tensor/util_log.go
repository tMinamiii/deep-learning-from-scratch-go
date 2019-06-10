package tensor

import (
	"github.com/naronA/zero_deeplearning/vec"
)

func Log(m *Tensor) *Tensor {
	if len(m.Shape) == 2 {
		mat := m.Mat
		return &Tensor{
			Mat:   logMat(mat),
			Shape: m.Shape,
		}
	}
	if len(m.Shape) == 3 {
		t3d := m.T3D
		return &Tensor{
			T3D:   logT3D(t3d),
			Shape: m.Shape,
		}
	}
	if len(m.Shape) == 4 {
		t4d := m.T4D
		return &Tensor{
			T4D:   logT4D(t4d),
			Shape: m.Shape,
		}
	}
	return nil
}

func logMat(m *Matrix) *Matrix {
	mat := vec.Log(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func logT3D(m Tensor3D) Tensor3D {
	t3d := make([]*Matrix, len(m))
	for i, mat := range t3d {
		t3d[i] = logMat(mat)
	}
	return t3d
}

func logT4D(m Tensor4D) Tensor4D {
	t4d := make([]Tensor3D, len(m))
	for i, t3d := range t4d {
		t4d[i] = logT3D(t3d)
	}
	return t4d
}
