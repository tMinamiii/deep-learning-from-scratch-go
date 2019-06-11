package tensor

import (
	"github.com/naronA/zero_deeplearning/vec"
)

type Tensor struct {
	Val   float64
	Vec   vec.Vector
	Mat   *Matrix
	T3D   Tensor3D
	T4D   Tensor4D
	T5D   Tensor5D
	T6D   Tensor6D
	Shape []int
}

func (t *Tensor) Size() int {
	size := 1
	for _, v := range t.Shape {
		size *= v
	}
	return size
}

func NewMatrix(row int, column int, vec vec.Vector) *Tensor {
	return &Tensor{
		Mat: &Matrix{
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
	mat := &Matrix{
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
	t3d := make(Tensor3D, c)
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
	t4d := make(Tensor4D, n)
	for i := 0; i < n; i++ {
		t4d[i] = NewRandnT3D(c, h, w).T3D
	}
	return &Tensor{
		T4D:   t4d,
		Shape: []int{c, h, w},
	}
}

func Zeros(shape []int) *Tensor {
	switch len(shape) {
	case 2:
		return &Tensor{Mat: zerosMat(shape), Shape: shape}
	case 3:
		return &Tensor{T3D: zerosT3D(shape), Shape: shape}
	case 4:
		return &Tensor{T4D: zerosT4D(shape), Shape: shape}
	case 5:
		return &Tensor{T5D: zerosT5D(shape), Shape: shape}
	case 6:
		return &Tensor{T6D: zerosT6D(shape), Shape: shape}
	}
	panic(shape)
}

func ZerosLike(t *Tensor) *Tensor {
	return Zeros(t.Shape)
}

func (t *Tensor) Ndim() int {
	return len(t.Shape)
}

func (t *Tensor) Window(x, y, h, w int) *Tensor {
	if len(t.Shape) == 2 {
		return &Tensor{
			Mat:   t.Mat.window(x, y, h, w),
			Shape: t.Shape,
		}
	}
	panic(t)
}

func (t *Tensor) Transpose(axis []int) *Tensor {
	switch {
	case len(t.Shape) == 2:
		trans := t.Mat.transpose(axis[0], axis[1])
		r, c := trans.Shape()
		shape := []int{r, c}
		return &Tensor{
			Mat:   trans,
			Shape: shape,
		}
	case len(t.Shape) == 3:
		trans := t.T3D.transpose(axis[0], axis[1], axis[2])
		c, h, w := trans.Shape()
		return &Tensor{
			T3D:   trans,
			Shape: []int{c, h, w},
		}
	case len(t.Shape) == 4:
		trans := t.T4D.transpose(axis[0], axis[1], axis[2], axis[3])
		n, c, h, w := trans.Shape()
		return &Tensor{
			T4D:   trans,
			Shape: []int{n, c, h, w},
		}
	case len(t.Shape) == 5:
		trans := t.T5D.transpose(axis[0], axis[1], axis[2], axis[3], axis[4])
		a, n, c, h, w := trans.Shape()
		return &Tensor{
			T5D:   trans,
			Shape: []int{a, n, c, h, w},
		}
	case len(t.Shape) == 6:
		trans := t.T6D.transpose(axis[0], axis[1], axis[2], axis[3], axis[4], axis[5])
		a, b, c, d, e, f := trans.Shape()
		return &Tensor{
			T6D:   trans,
			Shape: []int{a, b, c, d, e, f},
		}
	}
	panic(t)
}

/* Dependent */
func (t *Tensor) Element(point []int) float64 {
	switch len(t.Shape) {
	case 2:
		return t.Mat.element(point)
	case 3:
		return t.T3D.element(point)
	case 4:
		return t.T4D.element(point)
	case 5:
		return t.T5D.element(point)
	case 6:
		return t.T6D.element(point)
	}
	panic(t)
}

/* Depenedent*/
func (t *Tensor) Assign(value float64, point []int) {
	switch {
	case len(t.Shape) == 2:
		t.Mat.assign(value, point)
	case len(t.Shape) == 3:
		t.T3D.assign(value, point)
	case len(t.Shape) == 4:
		t.T4D.assign(value, point)
	case len(t.Shape) == 5:
		t.T5D.assign(value, point)
	case len(t.Shape) == 6:
		t.T6D.assign(value, point)
	}
	panic(t)
}

func (t *Tensor) AssignWindow(window *Matrix, x, y, h, w int) {
	if len(t.Shape) == 2 {
		t.Mat.assignWindow(window, x, y, h, w)
	}
	panic(t)
}

func AddAssign(t1 *Tensor4DSlice, t2 *Tensor) {
	if len(t2.Shape) == 4 {
		addAssignT4D(t1, t2.T4D)
	}
	panic(t2)
}
func (t *Tensor) Pad(pad int) *Tensor {
	switch len(t.Shape) {
	case 2:
		return &Tensor{Mat: t.Mat.pad(pad), Shape: t.Shape}
	case 3:
		return &Tensor{T3D: t.T3D.pad(pad), Shape: t.Shape}
	case 4:
		return &Tensor{T4D: t.T4D.pad(pad), Shape: t.Shape}
	case 5:
		return &Tensor{T5D: t.T5D.pad(pad), Shape: t.Shape}
	case 6:
		return &Tensor{T6D: t.T6D.pad(pad), Shape: t.Shape}
	}
	panic(t)
}

func (t *Tensor) Abs() *Tensor {
	switch len(t.Shape) {
	case 2:
		return &Tensor{Mat: t.Mat.abs(), Shape: t.Shape}
	case 3:
		return &Tensor{T3D: t.T3D.abs(), Shape: t.Shape}
	case 4:
		return &Tensor{T4D: t.T4D.abs(), Shape: t.Shape}
	}
	panic(t)
}

func (t *Tensor) ArgMaxAll() int {
	switch len(t.Shape) {
	case 2:
		return t.Mat.argMaxAll()
	case 3:
		return t.T3D.argMaxAll()
	case 4:
		return t.T4D.argMaxAll()
	}
	panic(t)
}

func (t *Tensor) ArgMax(axis int) []int {
	switch len(t.Shape) {
	case 2:
		return t.Mat.argMax(axis)
	}
	panic(t)
}

func (t *Tensor) CrossEntropyError(x *Tensor) float64 {
	switch {
	case len(t.Shape) == 2 && len(x.Shape) == 2:
		return t.Mat.crossEntropyError(x.Mat)
	case len(t.Shape) == 3 && len(x.Shape) == 3:
		return t.T3D.crossEntropyError(x.T3D)
	case len(t.Shape) == 4 && len(x.Shape) == 4:
		return t.T4D.crossEntropyError(x.T4D)
	}
	panic([]*Tensor{t, x})
}

type Arithmetic int

const (
	ADD Arithmetic = iota
	SUB
	MUL
	DIV
)

func Add(x1, x2 *Tensor) *Tensor {
	return calcArithmetic(ADD, x1, x2)
}

func Sub(x1, x2 *Tensor) *Tensor {
	return calcArithmetic(SUB, x1, x2)
}

func Mul(x1, x2 *Tensor) *Tensor {
	return calcArithmetic(MUL, x1, x2)
}

func Div(x1, x2 *Tensor) *Tensor {
	return calcArithmetic(DIV, x1, x2)
}

func calcArithmetic(a Arithmetic, x1, x2 *Tensor) *Tensor {
	if len(x1.Shape) == 4 {
		x1v := x1.T4D
		switch len(x2.Shape) {
		case 4:
			x2v := x2.T4D
			return &Tensor{T4D: t4DT4D(a, x1v, x2v), Shape: x1.Shape}
		case 3:
			x2v := x2.T3D
			return &Tensor{T4D: t4DT3D(a, x1v, x2v), Shape: x1.Shape}
		case 2:
			x2v := x2.Mat
			return &Tensor{T4D: t4DMat(a, x1v, x2v), Shape: x1.Shape}
		case 1:
			x2v := x2.Vec
			return &Tensor{T4D: t4DVec(a, x1v, x2v), Shape: x1.Shape}
		case 0:
			x2v := x2.Val
			return &Tensor{T4D: t4DFloat(a, x1v, x2v), Shape: x1.Shape}
		}
	} else if len(x2.Shape) == 4 {
		x2v := x2.T4D
		switch len(x1.Shape) {
		case 3:
			x1v := x1.T3D
			return &Tensor{T4D: t3DT4D(a, x1v, x2v), Shape: x1.Shape}
		case 2:
			x1v := x1.Mat
			return &Tensor{T4D: matT4D(a, x1v, x2v), Shape: x1.Shape}
		case 1:
			x1v := x1.Vec
			return &Tensor{T4D: vecT4D(a, x1v, x2v), Shape: x1.Shape}
		case 0:
			x1v := x1.Val
			return &Tensor{T4D: floatT4D(a, x1v, x2v), Shape: x1.Shape}
		}
	}
	if len(x1.Shape) == 3 {
		x1v := x1.T3D
		switch len(x2.Shape) {
		case 3:
			x2v := x2.T3D
			return &Tensor{T3D: t3DT3D(a, x1v, x2v), Shape: x1.Shape}
		case 2:
			x2v := x2.Mat
			return &Tensor{T3D: t3DMat(a, x1v, x2v), Shape: x1.Shape}
		case 1:
			x2v := x2.Vec
			return &Tensor{T3D: t3DVec(a, x1v, x2v), Shape: x1.Shape}
		case 0:
			x2v := x2.Val
			return &Tensor{T3D: t3DFloat(a, x1v, x2v), Shape: x1.Shape}
		}
	} else if len(x2.Shape) == 3 {
		x2v := x2.T3D
		switch len(x1.Shape) {
		case 2:
			x1v := x1.Mat
			return &Tensor{T3D: matT3D(a, x1v, x2v), Shape: x1.Shape}
		case 1:
			x1v := x1.Vec
			return &Tensor{T3D: vecT3D(a, x1v, x2v), Shape: x1.Shape}
		case 0:
			x1v := x1.Val
			return &Tensor{T3D: floatT3D(a, x1v, x2v), Shape: x1.Shape}
		}
	}
	if len(x1.Shape) == 2 {
		x1v := x1.Mat
		switch len(x2.Shape) {
		case 2:
			x2v := x2.Mat
			return &Tensor{Mat: matMat(ADD, x1v, x2v), Shape: x1.Shape}
		case 1:
			x2v := x2.Vec
			return &Tensor{Mat: matVec(ADD, x1v, x2v), Shape: x1.Shape}
		case 0:
			x2v := x2.Val
			return &Tensor{Mat: matFloat(ADD, x1v, x2v), Shape: x1.Shape}
		}
	} else if len(x2.Shape) == 2 {
		x2v := x2.Mat
		switch len(x1.Shape) {
		case 1:
			x1v := x1.Vec
			return &Tensor{Mat: vecMat(ADD, x1v, x2v), Shape: x1.Shape}
		case 0:
			x1v := x1.Val
			return &Tensor{Mat: floatMat(ADD, x1v, x2v), Shape: x1.Shape}
		}
	}
	panic([]*Tensor{x1, x2})
}

func (t *Tensor) NotEqual(x *Tensor) bool {
	return !t.Equal(x)
}

func (t *Tensor) Equal(x *Tensor) bool {
	if len(t.Shape) != len(x.Shape) {
		return false
	}
	switch len(t.Shape) {
	case 1:
		vec.Equal(t.Vec, x.Vec)
	case 2:
		t.Mat.equal(x.Mat)
	case 3:
		t.T3D.equal(x.T3D)
	case 4:
		t.T4D.equal(x.T4D)
	case 5:
		t.T5D.equal(x.T5D)
	case 6:
		t.T6D.equal(x.T6D)
	}
	return false
}

func (t *Tensor) Exp() *Tensor {
	switch len(t.Shape) {
	case 2:
		return &Tensor{Mat: t.Mat.exp(), Shape: t.Shape}
	case 3:
		return &Tensor{T3D: t.T3D.exp(), Shape: t.Shape}
	case 4:
		return &Tensor{T4D: t.T4D.exp(), Shape: t.Shape}
	}
	return nil
}

func (t *Tensor) Log() *Tensor {
	switch len(t.Shape) {
	case 2:
		return &Tensor{Mat: t.Mat.log(), Shape: t.Shape}
	case 3:
		return &Tensor{T3D: t.T3D.log(), Shape: t.Shape}
	case 4:
		return &Tensor{T4D: t.T4D.log(), Shape: t.Shape}
	}
	panic(t)
}

func (t *Tensor) Max(axis int) *Tensor {
	if len(t.Shape) == 2 {
		vec := t.Mat.max(axis)
		return &Tensor{Vec: vec, Shape: []int{len(vec)}}
	}
	panic(t)
}

func (t *Tensor) MaxAll() float64 {
	switch len(t.Shape) {
	case 2:
		return t.Mat.maxAll()
	case 3:
		return t.T3D.maxAll()
	case 4:
		return t.T4D.maxAll()
	}
	panic(t)
}

func (t *Tensor) MeanAll() float64 {
	switch len(t.Shape) {
	case 2:
		return t.Mat.meanAll()
	case 3:
		return t.T3D.meanAll()
	case 4:
		return t.T4D.meanAll()
	}
	panic(t)
}

func (t *Tensor) Mean(axis int) *Tensor {
	if len(t.Shape) == 2 {
		return &Tensor{Mat: t.Mat.mean(axis), Shape: t.Shape}
	}
	panic(t)
}

func (t *Tensor) Pow(p float64) *Tensor {
	if len(t.Shape) == 2 {
		return &Tensor{Mat: t.Mat.pow(p), Shape: t.Shape}
	}
	if len(t.Shape) == 3 {
		return &Tensor{T3D: t.T3D.pow(p), Shape: t.Shape}
	}
	if len(t.Shape) == 4 {
		return &Tensor{T4D: t.T4D.pow(p), Shape: t.Shape}
	}
	return nil
}
