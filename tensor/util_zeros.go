package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func zerosMat(shape []int) *types.Matrix {
	rows := shape[0]
	cols := shape[1]
	zeros := vec.Zeros(rows * cols)
	return &types.Matrix{
		Vector:  zeros,
		Rows:    rows,
		Columns: cols,
	}
}

func zerosT3D(shape []int) (t3d types.Tensor3D) {
	t3d = make(types.Tensor3D, shape[0])
	for i := range t3d {
		t3d[i] = zerosMat(shape[1:])
	}
	return
}

func zerosT4D(shape []int) (t4d types.Tensor4D) {
	t4d = make(types.Tensor4D, shape[0])
	for i := range t4d {
		t4d[i] = zerosT3D(shape[1:])
	}
	return
}

func zerosT5D(shape []int) (t5d types.Tensor5D) {
	t5d = make(types.Tensor5D, shape[0])
	for i := range t5d {
		t5d[i] = zerosT4D(shape[1:])
	}
	return
}

func zerosT6D(shape []int) (t6d types.Tensor6D) {
	t6d = make(types.Tensor6D, shape[0])
	for i := range t6d {
		t6d[i] = zerosT5D(shape[1:])
	}
	return t6d

}

func Zeros(shape []int) *Tensor {
	switch len(shape) {
	case 2:
		return &Tensor{
			Mat:   zerosMat(shape),
			Shape: shape,
		}
	case 3:
		return &Tensor{
			T3D:   zerosT3D(shape),
			Shape: shape,
		}
	case 4:
		return &Tensor{
			T4D:   zerosT4D(shape),
			Shape: shape,
		}
	case 5:
		return &Tensor{
			T5D:   zerosT5D(shape),
			Shape: shape,
		}
	case 6:
		return &Tensor{
			T6D:   zerosT6D(shape),
			Shape: shape,
		}
	}
	panic(shape)
}

func ZerosLike(t *Tensor) *Tensor {
	return Zeros(t.Shape)
}
