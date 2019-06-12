package tensor

import (
	"github.com/naronA/zero_deeplearning/vec"
)

type Tensor4D []Tensor3D

func (t Tensor4D) Size() int {
	n, c, h, w := t.Shape()
	return n * c * h * w
}

func (t Tensor4D) Element(n, c, h, w int) float64 {
	return t[n].Element(c, h, w)
}

func (t Tensor4D) Assign(value float64, n, c, h, w int) {
	t[n].Assign(value, c, h, w)
}

func (t Tensor4D) Flatten() vec.Vector {
	v := make(vec.Vector, 0, len(t)*len(t[0].Flatten()))
	for _, e := range t {
		v = append(v, e.Flatten()...)
	}
	return v
}
func (t Tensor4D) Shape() (int, int, int, int) {
	N := len(t)
	C := t[0].Channels()
	H, W := t[0][0].Shape()
	return N, C, H, W
}

func ZerosT4D(n, c, h, w int) Tensor4D {
	t4d := make(Tensor4D, n)
	for i := range t4d {
		t4d[i] = ZerosT3D(c, h, w)
	}
	return t4d
}

func ZerosLikeT4D(x Tensor4D) Tensor4D {
	t4d := make(Tensor4D, len(x))
	for i, v := range x {
		t4d[i] = ZerosLikeT3D(v)
	}
	return t4d
}

type Tensor4DIndex struct {
	N int
	C int
	H int
	W int
}

type Tensor4DSlice struct {
	Actual   Tensor4D
	Indices  []*Tensor4DIndex
	NewShape []int
}

func (t4s *Tensor4DSlice) ToTensor4D() Tensor4D {
	newT4D := ZerosT4D(t4s.NewShape[0], t4s.NewShape[1], t4s.NewShape[2], t4s.NewShape[3])
	for i, idx := range t4s.Indices {
		val := t4s.Actual[idx.N][idx.C].Element(idx.H, idx.W)
		matrixLength := t4s.NewShape[2] * t4s.NewShape[3]
		newMatIdx := i - idx.C*matrixLength - idx.N*(matrixLength*t4s.NewShape[1])
		newT4D[idx.N][idx.C].Vector[newMatIdx] = val
	}
	return newT4D
}

func (t Tensor4D) window(x, y, h, w int) Tensor4D {
	newT4D := make(Tensor4D, len(t))
	for i, t3d := range t {
		newT4D[i] = t3d.window(x, y, h, w)
	}
	return newT4D
}

func (t Tensor4D) transpose(a, b, c, d int) Tensor4D {
	w, x, y, z := t.Shape()
	shape := []int{w, x, y, z}
	t4d := zerosT4D([]int{shape[a], shape[b], shape[c], shape[d]})
	for i, e1t3d := range t {
		for j, e2mat := range e1t3d {
			for k := 0; k < e2mat.Rows; k++ {
				for l := 0; l < e2mat.Columns; l++ {
					oldIdx := []int{i, j, k, l}
					idx := make([]int, 4)
					idx[0] = oldIdx[a]
					idx[1] = oldIdx[b]
					idx[2] = oldIdx[c]
					idx[3] = oldIdx[d]
					v := t.element([]int{i, j, k, l})
					t4d.assign(v, []int{idx[0], idx[1], idx[2], idx[3]})
				}
			}
		}
	}
	return t4d
}

func (t Tensor4D) element(point []int) float64 {
	a := point[0]
	return t[a].element(point[1:])
}

func (t Tensor4D) assign(value float64, point []int) {
	a := point[0]
	t[a].assign(value, point[1:])
}

func addAssignT4D(t1 *Tensor4DSlice, t2 Tensor4D) {
	t2flat := t2.Flatten()
	for i, idx := range t1.Indices {
		add := t1.Actual[idx.N][idx.C].Element(idx.H, idx.W) + t2flat[i]
		t1.Actual[idx.N][idx.C].Assign(add, idx.H, idx.W)
	}
}

func zerosT4D(shape []int) (t4d Tensor4D) {
	t4d = make(Tensor4D, shape[0])
	for i := range t4d {
		t4d[i] = zerosT3D(shape[1:])
	}
	return
}

func (t Tensor4D) pad(size int) Tensor4D {
	newT4D := make(Tensor4D, len(t))
	for i, t3d := range t {
		padded := t3d.pad(size)
		newT4D[i] = padded
	}
	return newT4D
}

func (t Tensor4D) abs() Tensor4D {
	t4d := make([]Tensor3D, len(t))
	for i, t3d := range t4d {
		t4d[i] = t3d.abs()
	}
	return t4d
}

func (t Tensor4D) argMaxAll() int {
	max := 0
	for _, t3d := range t {
		max += t3d.argMaxAll()
	}
	return max
}

func (t Tensor4D) crossEntropyError(x Tensor4D) float64 {
	r := vec.Zeros(len(t))
	for i := range t {
		r[i] = t[i].crossEntropyError(x[i])
	}
	return vec.Sum(r) / float64(len(t))
}

func (t Tensor4D) equal(x Tensor4D) bool {
	for i := range t {
		if !t[i].equal(x[i]) {
			return false
		}
	}
	return true
}

func t4DT4D(a Arithmetic, x1 Tensor4D, x2 Tensor4D) Tensor4D {
	t4d := make(Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DT3D(a, x1[i], x2[i])
	}
	return t4d
}

func t4DT3D(a Arithmetic, x1 Tensor4D, x2 Tensor3D) Tensor4D {
	t4d := make(Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DT3D(a, x1[i], x2)
	}
	return t4d
}

func t4DMat(a Arithmetic, x1 Tensor4D, x2 *Matrix) Tensor4D {
	t4d := make(Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DMat(a, x1[i], x2)
	}
	return t4d
}

func t4DVec(a Arithmetic, x1 Tensor4D, x2 vec.Vector) Tensor4D {
	t4d := make(Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DVec(a, x1[i], x2)
	}
	return t4d
}

func t4DFloat(a Arithmetic, x1 Tensor4D, x2 float64) Tensor4D {
	t4d := make(Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DFloat(a, x1[i], x2)
	}
	return t4d
}

func t3DT4D(a Arithmetic, x1 Tensor3D, x2 Tensor4D) Tensor4D {
	return t4DT3D(a, x2, x1)
}

func matT4D(a Arithmetic, x1 *Matrix, x2 Tensor4D) Tensor4D {
	return t4DMat(a, x2, x1)
}

func vecT4D(a Arithmetic, x1 vec.Vector, x2 Tensor4D) Tensor4D {
	return t4DVec(a, x2, x1)
}

func floatT4D(a Arithmetic, x1 float64, x2 Tensor4D) Tensor4D {
	return t4DFloat(a, x2, x1)
}
func (t Tensor4D) exp() Tensor4D {
	t4d := make([]Tensor3D, len(t))
	for i, t3d := range t4d {
		t4d[i] = t3d.exp()
	}
	return t4d
}

func (t Tensor4D) log() Tensor4D {
	t4d := make([]Tensor3D, len(t))
	for i, t3d := range t4d {
		t4d[i] = t3d.log()
	}
	return t4d
}

func (t Tensor4D) maxAll() float64 {
	max := 0.0
	for _, t3d := range t {
		max += t3d.maxAll()
	}
	return max
}

func (t Tensor4D) meanAll() float64 {
	return t.sumAll() / float64(len(t))
}

func (t Tensor4D) pow(p float64) Tensor4D {
	t4d := make([]Tensor3D, len(t))
	for i, t3d := range t4d {
		t4d[i] = t3d.pow(p)
	}
	return t4d
}

func (t Tensor4D) sumAll() float64 {
	sum := 0.0
	for _, t3d := range t {
		sum += t3d.sumAll()
	}
	return sum
}

func (t Tensor4D) sqrt() Tensor4D {
	t4d := make([]Tensor3D, len(t))
	for i, t3d := range t4d {
		t4d[i] = t3d.sqrt()
	}
	return t4d
}

func (t Tensor4D) softmax() Tensor4D {
	t4d := make([]Tensor3D, len(t))
	for i, t3d := range t4d {
		t4d[i] = t3d.softmax()
	}
	return t4d
}

func (t Tensor4D) sigmoid() Tensor4D {
	t4d := make([]Tensor3D, len(t))
	for i, t3d := range t4d {
		t4d[i] = t3d.sigmoid()
	}
	return t4d
}

func (t Tensor4D) relu() Tensor4D {
	t4d := make([]Tensor3D, len(t))
	for i, t3d := range t4d {
		t4d[i] = t3d.relu()
	}
	return t4d
}

func (t Tensor4D) numericalGradient(f func(vec.Vector) float64) Tensor4D {
	result := make(Tensor4D, len(t))
	for i, v := range t {
		result[i] = v.numericalGradient(f)
	}
	return result
}
