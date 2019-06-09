package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func NotEqual(t1, t2 *Tensor) bool {
	return !Equal(t1, t2)
}

func equalMat(m1, m2 *types.Matrix) bool {
	if m1.Rows == m2.Rows &&
		m1.Columns == m2.Columns &&
		vec.Equal(m1.Vector, m2.Vector) {
		return true
	}
	return false
}

func equalT3D(t1, t2 types.Tensor3D) bool {
	for i := range t1 {
		if !equalMat(t1[i], t2[i]) {
			return false
		}
	}
	return true
}

func equalT4D(t1, t2 types.Tensor4D) bool {
	for i := range t1 {
		if !equalT3D(t1[i], t2[i]) {
			return false
		}
	}
	return true
}

func equalT5D(t1, t2 types.Tensor5D) bool {
	for i := range t1 {
		if !equalT4D(t1[i], t2[i]) {
			return false
		}
	}
	return true
}

func equalT6D(t1, t2 types.Tensor6D) bool {
	for i := range t1 {
		if !equalT5D(t1[i], t2[i]) {
			return false
		}
	}
	return true
}

func Equal(t1, t2 *Tensor) bool {
	if len(t1.Shape) != len(t2.Shape) {
		return false
	}
	switch len(t1.Shape) {
	case 1:
		vec.Equal(t1.Vec, t2.Vec)
	case 2:
		equalMat(t1.Mat, t2.Mat)
	case 3:
		equalT3D(t1.T3D, t2.T3D)
	case 4:
		equalT4D(t1.T4D, t2.T4D)
	case 5:
		equalT5D(t1.T5D, t2.T5D)
	case 6:
		equalT6D(t1.T6D, t2.T6D)
	}
	return false
}
