package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
)

func Softmax(m *Tensor) *Tensor {
	if len(m.Shape) == 2 {
		mat := m.Mat
		return &Tensor{
			Mat:   softmaxMat(mat),
			Shape: m.Shape,
		}
	}
	if len(m.Shape) == 3 {
		t3d := m.T3D
		return &Tensor{
			T3D:   softmaxT3D(t3d),
			Shape: m.Shape,
		}
	}
	if len(m.Shape) == 4 {
		t4d := m.T4D
		return &Tensor{
			T4D:   softmaxT4D(t4d),
			Shape: m.Shape,
		}
	}
	return nil
}

func softmaxMat(x *types.Matrix) *types.Matrix {
	xt := x.T()
	sub := types.Sub(xt, types.Max(xt, 0))
	expX := types.Exp(sub)
	sumExpX := types.Sum(expX, 0)
	softmax := types.Div(expX, sumExpX.Vector)
	return softmax.T()
}

func softmaxT3D(m types.Tensor3D) types.Tensor3D {
	t3d := make([]*types.Matrix, len(m))
	for i, mat := range t3d {
		t3d[i] = softmaxMat(mat)
	}
	return t3d
}

func softmaxT4D(m types.Tensor4D) types.Tensor4D {
	t4d := make([]types.Tensor3D, len(m))
	for i, t3d := range t4d {
		t4d[i] = softmaxT3D(t3d)
	}
	return t4d
}
