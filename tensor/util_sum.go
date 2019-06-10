package tensor

import (
	"github.com/naronA/zero_deeplearning/vec"
)

func Sum(t *Tensor, axis int) *Tensor {
	if len(t.Shape) == 2 {
		return &Tensor{
			Mat:   t.Mat.sum(axis),
			Shape: t.Shape,
		}
	}
	panic(t)
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

func sumAllMat(m *Matrix) float64 {
	return vec.Sum(m.Vector)
}

func sumAllT3D(m Tensor3D) float64 {
	sum := 0.0
	for _, mat := range m {
		sum += sumAllMat(mat)
	}
	return sum
}

func sumAllT4D(m Tensor4D) float64 {
	sum := 0.0
	for _, t3d := range m {
		sum += sumAllT3D(t3d)
	}
	return sum
}
