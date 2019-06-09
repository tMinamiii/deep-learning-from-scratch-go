package types

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

