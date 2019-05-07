package tensor

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

type Tensor3D struct {
	Matrixes []*mat.Matrix
	Channels int
}

func tenTen(a Arithmetic, x1 Tensor3D, x2 Tensor3D) *Tensor3D {
	return nil
}
func tenMat(a Arithmetic, x1 Tensor3D, x2 Tensor3D) *Tensor3D {
	return nil
}
func tenVec(a Arithmetic, x1 Tensor3D, x2 Tensor3D) *Tensor3D {
	return nil
}
func tenFloat(a Arithmetic, x1 Tensor3D, x2 Tensor3D) *Tensor3D {
	return nil
}

func matTen(a Arithmetic, x1 Tensor3D, x2 Tensor3D) *Tensor3D {
	return nil
}
func vetTen(a Arithmetic, x1 vec.Vector, x2 Tensor3D) *Tensor3D {
	return nil
}
func floatTen(a Arithmetic, x1 float64, x2 Tensor3D) *Tensor3D {
	return nil
}

type Arithmetic int

const (
	ADD Arithmetic = iota
	SUB
	MUL
	DIV
)

func Add(x1 interface{}, x2 interface{}) *Tensor3D {
	if x1v, ok := x1.(*Tensor3D); ok {
		switch x2v := x2.(type) {
		case *Tensor3D:
			return tenTen(ADD, x1v, x2v)
		case *mat.Matrix:
			return tenMat(ADD, x1v, x2v)
		case vec.Vector:
			return tenVec(ADD, x1v, x2v)
		case float64:
			return tenFloat(ADD, x1v, x2v)
		case int:
			return tenFloat(ADD, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(*Tensor3D); ok {
		switch x1v := x1.(type) {
		case mat.Matrix:
			return matTen(ADD, x1v, x2v)
		case vec.Vector:
			return vecTen(ADD, x1v, x2v)
		case float64:
			return floatTen(ADD, x1v, x2v)
		case int:
			return floatTen(ADD, float64(x1v), x2v)
		}
	}
	return nil
}
