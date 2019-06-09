package tensor

import "github.com/naronA/zero_deeplearning/tensor/types"

/* Dependent */
func (t *Tensor) Element(point []int) float64 {
	switch len(t.Shape) {
	case 2:
		m := t.Mat
		return elementMat(m, point)
	case 3:
		x := t.T3D
		return elementT3D(x, point)
	case 4:
		x := t.T4D
		return elementT4D(x, point)
	case 5:
		x := t.T5D
		return elementT5D(x, point)
	case 6:
		x := t.T6D
		return elementT6D(x, point)

	}
	panic(t)
}

func elementMat(x *types.Matrix, point []int) float64 {
	r := point[0]
	c := point[1]
	return x.Vector[r*x.Columns+c]
}

func elementT3D(x types.Tensor3D, point []int) float64 {
	a := point[0]
	return elementMat(x[a], point[1:])
}

func elementT4D(x types.Tensor4D, point []int) float64 {
	a := point[0]
	return elementT3D(x[a], point[1:])
}

func elementT5D(x types.Tensor5D, point []int) float64 {
	a := point[0]
	return elementT4D(x[a], point[1:])
}

func elementT6D(x types.Tensor6D, point []int) float64 {
	a := point[0]
	return elementT5D(x[a], point[1:])
}
