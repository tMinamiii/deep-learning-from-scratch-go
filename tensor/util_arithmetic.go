package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

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

func matMat(a Arithmetic, m1, m2 *types.Matrix) *types.Matrix {
	if m1.Rows != m2.Rows && m1.Columns != m2.Columns {
		// 片方がベクトル(1行多列)だった場合
		if m1.Rows == 1 || m2.Rows == 1 {
			if m1.Columns != m2.Columns {
				return nil
			}
			vector := vec.Zeros(m1.Rows * m1.Columns)
			for r := 0; r < m1.Rows; r++ {
				for c := 0; c < m2.Columns; c++ {
					index := r*m1.Columns + c
					switch a {
					case ADD:
						vector[index] = m1.Element(r, c) + m2.Element(0, c)
					case SUB:
						vector[index] = m1.Element(r, c) - m2.Element(0, c)
					case MUL:
						vector[index] = m1.Element(r, c) * m2.Element(0, c)
					case DIV:
						vector[index] = m1.Element(r, c) / m2.Element(0, c)
					}
				}
			}
			return &types.Matrix{
				Vector:  vector,
				Rows:    m1.Rows,
				Columns: m1.Columns,
			}

		}
	} else {
		vector := vec.Zeros(m1.Rows * m1.Columns)
		switch a {
		case ADD:
			vector = vec.Add(m1.Vector, m2.Vector)
		case SUB:
			vector = vec.Sub(m1.Vector, m2.Vector)
		case MUL:
			vector = vec.Mul(m1.Vector, m2.Vector)
		case DIV:
			vector = vec.Div(m1.Vector, m2.Vector)
		}

		return &types.Matrix{
			Vector:  vector,
			Rows:    m1.Rows,
			Columns: m1.Columns,
		}
	}
	panic([]*types.Matrix{m1, m2})
}

func matVec(a Arithmetic, m1 *types.Matrix, m2 vec.Vector) *types.Matrix {
	if m1.Columns != len(m2) {
		return nil
	}
	vector := vec.Zeros(m1.Rows * m1.Columns)
	for r := 0; r < m1.Rows; r++ {
		for c := 0; c < len(m2); c++ {
			index := r*m1.Columns + c
			switch a {
			case ADD:
				vector[index] = m1.Element(r, c) + m2[c]
			case SUB:
				vector[index] = m1.Element(r, c) - m2[c]
			case MUL:
				vector[index] = m1.Element(r, c) * m2[c]
			case DIV:
				vector[index] = m1.Element(r, c) / m2[c]
			}
		}
	}
	return &types.Matrix{
		Vector:  vector,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}

func matFloat(a Arithmetic, m1 *types.Matrix, m2 float64) *types.Matrix {
	vector := vec.ZerosLike(m1.Vector)
	switch a {
	case ADD:
		vector = vec.Add(m1.Vector, m2)
	case SUB:
		vector = vec.Sub(m1.Vector, m2)
	case MUL:
		vector = vec.Mul(m1.Vector, m2)
	case DIV:
		vector = vec.Div(m1.Vector, m2)
	}
	return &types.Matrix{
		Vector:  vector,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}

func vecMat(a Arithmetic, m1 vec.Vector, m2 *types.Matrix) *types.Matrix {
	if m2.Columns != len(m1) {
		return nil
	}
	vector := vec.Zeros(m2.Rows * m2.Columns)
	for r := 0; r < m2.Rows; r++ {
		for c := 0; c < len(m1); c++ {
			index := r*m2.Columns + c
			switch a {
			case ADD:
				vector[index] = m1[c] + m2.Element(r, c)
			case SUB:
				vector[index] = m1[c] - m2.Element(r, c)
			case MUL:
				vector[index] = m1[c] * m2.Element(r, c)
			case DIV:
				vector[index] = m1[c] / m2.Element(r, c)
			}
		}
	}
	return &types.Matrix{
		Vector:  vector,
		Rows:    m2.Rows,
		Columns: m2.Columns,
	}
}

func floatMat(a Arithmetic, m1 float64, m2 *types.Matrix) *types.Matrix {
	vector := vec.ZerosLike(m2.Vector)
	switch a {
	case ADD:
		vector = vec.Add(m1, m2.Vector)
	case SUB:
		vector = vec.Sub(m1, m2.Vector)
	case MUL:
		vector = vec.Mul(m1, m2.Vector)
	case DIV:
		vector = vec.Div(m1, m2.Vector)
	}
	return &types.Matrix{
		Vector:  vector,
		Rows:    m2.Rows,
		Columns: m2.Columns,
	}
}

func t3DT3D(a Arithmetic, x1, x2 types.Tensor3D) types.Tensor3D {
	t3d := make(types.Tensor3D, len(x1))
	for i := range x1 {
		t3d[i] = matMat(a, x1[i], x2[i])
	}
	return t3d
}

func t3DMat(a Arithmetic, x1 types.Tensor3D, x2 *types.Matrix) types.Tensor3D {
	t3d := make(types.Tensor3D, len(x1))
	for i := range x1 {
		t3d[i] = matMat(a, x1[i], x2)
	}
	return t3d
}

func t3DVec(a Arithmetic, x1 types.Tensor3D, x2 vec.Vector) types.Tensor3D {
	t3d := make(types.Tensor3D, len(x1))
	for i := range x1 {
		t3d[i] = matVec(a, x1[i], x2)
	}
	return t3d
}

func t3DFloat(a Arithmetic, x1 types.Tensor3D, x2 float64) types.Tensor3D {
	t3d := make(types.Tensor3D, len(x1))
	for i := range x1 {
		t3d[i] = matFloat(a, x1[i], x2)
	}
	return t3d
}

func matT3D(a Arithmetic, x1 *types.Matrix, x2 types.Tensor3D) types.Tensor3D {
	return t3DMat(a, x2, x1)
}

func vecT3D(a Arithmetic, x1 vec.Vector, x2 types.Tensor3D) types.Tensor3D {
	return t3DVec(a, x2, x1)
}

func floatT3D(a Arithmetic, x1 float64, x2 types.Tensor3D) types.Tensor3D {
	return t3DFloat(a, x2, x1)
}

func t4DT4D(a Arithmetic, x1 types.Tensor4D, x2 types.Tensor4D) types.Tensor4D {
	t4d := make(types.Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DT3D(a, x1[i], x2[i])
	}
	return t4d
}

func t4DT3D(a Arithmetic, x1 types.Tensor4D, x2 types.Tensor3D) types.Tensor4D {
	t4d := make(types.Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DT3D(a, x1[i], x2)
	}
	return t4d
}

func t4DMat(a Arithmetic, x1 types.Tensor4D, x2 *types.Matrix) types.Tensor4D {
	t4d := make(types.Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DMat(a, x1[i], x2)
	}
	return t4d
}

func t4DVec(a Arithmetic, x1 types.Tensor4D, x2 vec.Vector) types.Tensor4D {
	t4d := make(types.Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DVec(a, x1[i], x2)
	}
	return t4d
}

func t4DFloat(a Arithmetic, x1 types.Tensor4D, x2 float64) types.Tensor4D {
	t4d := make(types.Tensor4D, len(x1))
	for i := range x1 {
		t4d[i] = t3DFloat(a, x1[i], x2)
	}
	return t4d
}

func t3DT4D(a Arithmetic, x1 types.Tensor3D, x2 types.Tensor4D) types.Tensor4D {
	return t4DT3D(a, x2, x1)
}

func matT4D(a Arithmetic, x1 *types.Matrix, x2 types.Tensor4D) types.Tensor4D {
	return t4DMat(a, x2, x1)
}

func vecT4D(a Arithmetic, x1 vec.Vector, x2 types.Tensor4D) types.Tensor4D {
	return t4DVec(a, x2, x1)
}

func floatT4D(a Arithmetic, x1 float64, x2 types.Tensor4D) types.Tensor4D {
	return t4DFloat(a, x2, x1)
}
