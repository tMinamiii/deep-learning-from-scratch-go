package tensor

import (
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

func powMat(m *Matrix, p float64) *Matrix {
	mat := vec.Pow(m.Vector, p)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func powT3D(m Tensor3D, p float64) Tensor3D {
	t3d := make([]*Matrix, len(m))
	for i, mat := range t3d {
		t3d[i] = powMat(mat, p)
	}
	return t3d
}

func powT4D(m Tensor4D, p float64) Tensor4D {
	t4d := make([]Tensor3D, len(m))
	for i, t3d := range t4d {
		t4d[i] = powT3D(t3d, p)
	}
	return t4d
}
