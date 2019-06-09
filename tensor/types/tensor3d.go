package types

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
