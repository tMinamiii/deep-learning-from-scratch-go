package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func CrossEntropyError(y, t *Tensor) float64 {
	if len(y.Shape) == 2 && len(t.Shape) == 2 {
		return crossEntropyErrorMat(y.Mat, t.Mat)
	} else if len(y.Shape) == 3 && len(t.Shape) == 3 {
		return crossEntropyErrorT3D(y.T3D, t.T3D)
	} else if len(y.Shape) == 4 && len(t.Shape) == 4 {
		return crossEntropyErrorT4D(y.T4D, t.T4D)
	}
	panic(0)
}

func crossEntropyErrorMat(y, t *types.Matrix) float64 {
	r := vec.Zeros(y.Rows)
	for i := 0; i < y.Rows; i++ {
		yRow := y.SliceRow(i)
		tRow := t.SliceRow(i)
		r[i] = vec.CrossEntropyError(yRow, tRow)
	}
	return vec.Sum(r) / float64(y.Rows)
}
func crossEntropyErrorT3D(y, t types.Tensor3D) float64 {
	r := vec.Zeros(len(y))
	for i := range y {
		r[i] = crossEntropyErrorMat(y[i], t[i])
	}
	return vec.Sum(r) / float64(len(y))
}

func crossEntropyErrorT4D(y, t types.Tensor4D) float64 {
	r := vec.Zeros(len(y))
	for i := range y {
		r[i] = crossEntropyErrorT3D(y[i], t[i])
	}
	return vec.Sum(r) / float64(len(y))
}
