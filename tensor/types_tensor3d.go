package tensor

import (
	"github.com/naronA/zero_deeplearning/vec"
)

type Tensor3D []*Matrix

func (t Tensor3D) Flatten() vec.Vector {
	v := make(vec.Vector, 0, len(t)*len(t[0].Vector))
	for _, e := range t {
		v = append(v, e.Vector...)
	}
	return v
}

func (t Tensor3D) Channels() int {
	return len(t)
}

func (t Tensor3D) Element(c, h, w int) float64 {
	return t[c].Element(h, w)
}

func (t Tensor3D) Assign(value float64, c, h, w int) {
	t[c].Assign(value, h, w)
}

func (t Tensor3D) Shape() (int, int, int) {
	C := t.Channels()
	H, W := t[0].Shape()
	return C, H, W
}

func ZerosT3D(c, h, w int) Tensor3D {
	t3d := make(Tensor3D, c)
	for i := range t3d {
		t3d[i] = ZerosMat(h, w)
	}
	return t3d
}

func ZerosLikeT3D(x Tensor3D) Tensor3D {
	matrixes := make(Tensor3D, len(x))
	for i, v := range x {
		matrixes[i] = ZerosLikeMat(v)
	}
	return matrixes
}

func (t Tensor3D) window(x, y, h, w int) Tensor3D {
	newT3D := make(Tensor3D, len(t))
	for i, mat := range t {
		newT3D[i] = mat.window(x, y, h, w)
	}
	return newT3D
}

func (t Tensor3D) transpose(a, b, c int) Tensor3D {
	x, y, z := t.Shape()
	shape := []int{x, y, z}
	t3d := ZerosT3D(shape[a], shape[b], shape[c])
	for i, mat := range t {
		for j := 0; j < mat.Rows; j++ {
			for k := 0; k < mat.Columns; k++ {
				oldIdx := []int{i}
				idx := make([]int, 3)
				idx[0] = oldIdx[a]
				idx[1] = oldIdx[b]
				idx[2] = oldIdx[c]
				v := t.element([]int{i, j, k})
				t3d.assign(v, []int{idx[0], idx[1], idx[2]})
			}
		}
	}
	return t3d
}

func (t Tensor3D) element(point []int) float64 {
	a := point[0]
	return t[a].element(point[1:])
}

func (t Tensor3D) assign(value float64, point []int) {
	a := point[0]
	t[a].assign(value, point[1:])
}

func zerosT3D(shape []int) (t3d Tensor3D) {
	t3d = make(Tensor3D, shape[0])
	for i := range t3d {
		t3d[i] = zerosMat(shape[1:])
	}
	return
}

func (t Tensor3D) pad(size int) Tensor3D {
	newT3D := make(Tensor3D, len(t))
	for i, m := range t {
		newT3D[i] = m.pad(size)
	}
	return newT3D
}

func (t Tensor3D) abs() Tensor3D {
	t3d := make([]*Matrix, len(t))
	for i, mat := range t3d {
		t3d[i] = mat.abs()
	}
	return t3d
}

func (t Tensor3D) argMaxAll() int {
	max := 0
	for _, mat := range t {
		max += mat.argMaxAll()
	}
	return max
}

func (t Tensor3D) crossEntropyError(x Tensor3D) float64 {
	r := vec.Zeros(len(t))
	for i := range t {
		r[i] = t[i].crossEntropyError(x[i])
	}
	return vec.Sum(r) / float64(len(t))
}

func (t Tensor3D) equal(x Tensor3D) bool {
	for i := range t {
		if !t[i].equal(x[i]) {
			return false
		}
	}
	return true
}

func t3DT3D(a Arithmetic, x1, x2 Tensor3D) Tensor3D {
	t3d := make(Tensor3D, len(x1))
	for i := range x1 {
		t3d[i] = matMat(a, x1[i], x2[i])
	}
	return t3d
}

func t3DMat(a Arithmetic, x1 Tensor3D, x2 *Matrix) Tensor3D {
	t3d := make(Tensor3D, len(x1))
	for i := range x1 {
		t3d[i] = matMat(a, x1[i], x2)
	}
	return t3d
}

func t3DVec(a Arithmetic, x1 Tensor3D, x2 vec.Vector) Tensor3D {
	t3d := make(Tensor3D, len(x1))
	for i := range x1 {
		t3d[i] = matVec(a, x1[i], x2)
	}
	return t3d
}

func t3DFloat(a Arithmetic, x1 Tensor3D, x2 float64) Tensor3D {
	t3d := make(Tensor3D, len(x1))
	for i := range x1 {
		t3d[i] = matFloat(a, x1[i], x2)
	}
	return t3d
}

func matT3D(a Arithmetic, x1 *Matrix, x2 Tensor3D) Tensor3D {
	return t3DMat(a, x2, x1)
}

func vecT3D(a Arithmetic, x1 vec.Vector, x2 Tensor3D) Tensor3D {
	return t3DVec(a, x2, x1)
}

func floatT3D(a Arithmetic, x1 float64, x2 Tensor3D) Tensor3D {
	return t3DFloat(a, x2, x1)
}

func (t Tensor3D) exp() Tensor3D {
	t3d := make([]*Matrix, len(t))
	for i, mat := range t3d {
		t3d[i] = mat.exp()
	}
	return t3d
}

func (t Tensor3D) log() Tensor3D {
	t3d := make([]*Matrix, len(t))
	for i, mat := range t3d {
		t3d[i] = mat.log()
	}
	return t3d
}

func (t Tensor3D) maxAll() float64 {
	max := 0.0
	for _, mat := range t {
		max += mat.maxAll()
	}
	return max
}

func (t Tensor3D) meanAll() float64 {
	return t.sumAll() / float64(len(t))
}

func (t Tensor3D) pow(p float64) Tensor3D {
	t3d := make([]*Matrix, len(t))
	for i, mat := range t3d {
		t3d[i] = mat.pow(p)
	}
	return t3d
}
func (t Tensor3D) sumAll() float64 {
	sum := 0.0
	for _, mat := range t {
		sum += mat.sumAll()
	}
	return sum
}

func (t Tensor3D) sqrt() Tensor3D {
	t3d := make([]*Matrix, len(t))
	for i, mat := range t3d {
		t3d[i] = mat.sqrt()
	}
	return t3d
}
func (t Tensor3D) softmax() Tensor3D {
	t3d := make([]*Matrix, len(t))
	for i, mat := range t3d {
		t3d[i] = mat.softmax()
	}
	return t3d
}

func (t Tensor3D) sigmoid() Tensor3D {
	t3d := make([]*Matrix, len(t))
	for i, mat := range t3d {
		t3d[i] = mat.sigmoid()
	}
	return t3d
}

func (t Tensor3D) relu() Tensor3D {
	t3d := make([]*Matrix, len(t))
	for i, mat := range t3d {
		t3d[i] = mat.relu()
	}
	return t3d
}

func (t Tensor3D) numericalGradient(f func(vec.Vector) float64) Tensor3D {
	result := make(Tensor3D, len(t))
	for i, v := range t {
		result[i] = v.numericalGradient(f)
	}
	return result
}
