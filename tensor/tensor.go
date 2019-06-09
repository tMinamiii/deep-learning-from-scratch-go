package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

type Tensor struct {
	Val   float64
	Vec   vec.Vector
	Mat   *types.Matrix
	T3D   types.Tensor3D
	T4D   types.Tensor4D
	T5D   types.Tensor5D
	T6D   types.Tensor6D
	Shape []int
}

func (t *Tensor) Size() int {
	size := 1
	for _, v := range t.Shape {
		size *= v
	}
	return size
}

func (t *Tensor) T() *Tensor {
	return t.Transpose([]int{1, 0})
}

func NewMatrix(row int, column int, vec vec.Vector) *Tensor {
	return &Tensor{
		Mat: &types.Matrix{
			Vector:  vec,
			Rows:    row,
			Columns: column,
		},
		Shape: []int{row, column},
	}
}

func NewRandnMatrix(row, column int) *Tensor {
	if row == 0 || column == 0 {
		panic(0)
	}
	vec := vec.Randn(row * column)
	mat := &types.Matrix{
		Vector:  vec,
		Rows:    row,
		Columns: column,
	}

	return &Tensor{
		Mat:   mat,
		Shape: []int{row, column},
	}
}

func NewRandnT3D(c, h, w int) *Tensor {
	if c == 0 || h == 0 || w == 0 {
		panic(0)
	}
	t3d := make(types.Tensor3D, c)
	for i := 0; i < c; i++ {
		mat := NewRandnMatrix(h, w).Mat
		t3d[i] = mat
	}
	return &Tensor{
		T3D:   t3d,
		Shape: []int{c, h, w},
	}
}

func NewRandnT4D(n, c, h, w int) *Tensor {
	if n == 0 || c == 0 || h == 0 || w == 0 {
		panic(0)
	}
	t4d := make(types.Tensor4D, n)
	for i := 0; i < n; i++ {
		t4d[i] = NewRandnT3D(c, h, w).T3D
	}
	return &Tensor{
		T4D:   t4d,
		Shape: []int{c, h, w},
	}
}
func (t *Tensor) Ndim() int {
	return len(t.Shape)
}
