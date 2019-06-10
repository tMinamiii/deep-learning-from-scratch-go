package tensor

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/vec"
)

type Matrix struct {
	Vector  vec.Vector
	Rows    int
	Columns int
}

func (m *Matrix) T() *Matrix {
	return m.transposeMat(1, 0)
}

func (m *Matrix) Shape() (int, int) {
	if m.Rows == 1 {
		return m.Columns, -1
	}
	return m.Rows, m.Columns
}

// Matrixなのでndimは常に2
func (m *Matrix) Ndim() int {
	return 2
}

func (m *Matrix) Element(r int, c int) float64 {
	return m.Vector[r*m.Columns+c]
}

func (m *Matrix) Assign(value float64, r, c int) {
	m.Vector[r*m.Columns+c] = value
}

func (m *Matrix) String() string {
	str := "[\n"
	for i := 0; i < m.Rows; i++ {
		str += fmt.Sprintf("  %v,\n", m.Vector[i*m.Columns:(i+1)*m.Columns])
	}
	str += "]"
	return str
}

func ZerosMat(r, c int) *Matrix {
	zeros := vec.Zeros(r * c)
	return &Matrix{
		Vector:  zeros,
		Rows:    r,
		Columns: c,
	}
}

func ZerosLikeMat(m *Matrix) *Matrix {
	return ZerosMat(m.Rows, m.Columns)
}

func (m *Matrix) SliceRow(r int) vec.Vector {
	slice := m.Vector[r*m.Columns : (r+1)*m.Columns]
	return slice
}

func (m *Matrix) SliceColumn(c int) vec.Vector {
	slice := vec.Zeros(m.Rows)
	for i := 0; i < m.Rows; i++ {
		slice[i] = m.Element(i, c)
	}
	return slice
}

func (m *Matrix) window(x, y, h, w int) *Matrix {
	mat := zerosMat([]int{h, w})
	for i := x; i < x+h; i++ {
		for j := y; j < y+w; j++ {
			mat.Vector[(i-x)*w+(j-y)] = m.Element(i, j)
		}
	}
	return mat
}

func (m *Matrix) transpose(a, b int) *Matrix {
	trans := make(vec.Vector, m.Rows*m.Columns)
	if a == 0 && b == 1 {
		for i := 0; i < m.Rows; i++ {
			col := m.SliceRow(i)
			for j := 0; j < len(col); j++ {
				trans[i*len(col)+j] = col[j]
			}
		}
		return &Matrix{
			Vector:  trans,
			Rows:    m.Rows,
			Columns: m.Columns,
		}

	}
	for i := 0; i < m.Columns; i++ {
		col := m.SliceColumn(i)
		// trans = append(trans, col...)
		for j := 0; j < len(col); j++ {
			trans[i*len(col)+j] = col[j]
		}
	}
	return &Matrix{
		Vector:  trans,
		Rows:    m.Columns,
		Columns: m.Rows,
	}
}

func (m *Matrix) element(point []int) float64 {
	r := point[0]
	c := point[1]
	return m.Vector[r*m.Columns+c]
}

func (m *Matrix) assign(value float64, point []int) {
	a := point[0]
	b := point[1]
	m.Vector[a*m.Columns+b] = value
}

func (m *Matrix) assignWindow(window *Matrix, x, y, h, w int) {
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			val := window.Element(i, j)
			m.assign(val, []int{i + x, j + y})
		}
	}
}

func zerosMat(shape []int) *Matrix {
	rows := shape[0]
	cols := shape[1]
	zeros := vec.Zeros(rows * cols)
	return &Matrix{
		Vector:  zeros,
		Rows:    rows,
		Columns: cols,
	}
}

func softmaxMat(x *Matrix) *Matrix {
	xt := x.T()
	sub := matVec(SUB, xt, xt.max(0))
	expX := Exp(sub)
	sumExpX := Sum(expX, 0)
	softmax := matVec(DIV, expX, sumExpX.Vector)
	return softmax.T()
}

func (m *Matrix) pad(pad int) *Matrix {
	if pad == 0 {
		return &Matrix{
			Vector:  m.Vector,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	}
	col := m.Columns
	newVec := make(vec.Vector, 0, m.Rows+2*pad)
	rowPad := vec.Zeros(col + 2*pad)
	for j := 0; j < pad; j++ {
		newVec = append(newVec, rowPad...)
	}
	for j := 0; j < m.Rows; j++ {
		srow := m.SliceRow(j)
		for k := 0; k < pad; k++ {
			newVec = append(newVec, 0)
		}
		newVec = append(newVec, srow...)
		for k := 0; k < pad; k++ {
			newVec = append(newVec, 0)
		}
	}
	for j := 0; j < pad; j++ {
		newVec = append(newVec, rowPad...)
	}
	return &Matrix{
		Vector:  newVec,
		Rows:    m.Rows + 2*pad,
		Columns: m.Columns + 2*pad,
	}
}

func (m *Matrix) max(axis int) vec.Vector {
	if axis == 0 {
		v := vec.Zeros(m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Max(col)
		}
		return v
	} else if axis == 1 {
		v := vec.Zeros(m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.SliceRow(i)
			v[i] = vec.Max(row)
		}
		return v
	}
	panic(m)
}

func (m *Matrix) sum(axis int) *Matrix {
	if axis == 0 {
		v := vec.Zeros(m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Sum(col)
		}
		return &Matrix{
			Vector:  v,
			Rows:    1,
			Columns: m.Columns,
		}
	} else if axis == 1 {
		v := vec.Zeros(m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.SliceRow(i)
			v[i] = vec.Sum(row)
		}
		return &Matrix{
			Vector:  v,
			Rows:    1,
			Columns: m.Rows,
		}
	}
	panic(m)
}

func (m *Matrix) abs() *Matrix {
	mat := vec.Abs(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Matrix) argMaxAll() int {
	return vec.ArgMax(m.Vector)
}

func (m *Matrix) argMax(axis int) []int {
	if axis == 0 {
		v := make([]int, m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.ArgMax(col)
		}
		return v
	} else if axis == 1 {
		v := make([]int, m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.SliceRow(i)
			v[i] = vec.ArgMax(row)
		}
		return v
	}
	panic(m)
}

func (m *Matrix) crossEntropyError(x *Matrix) float64 {
	r := vec.Zeros(m.Rows)
	for i := 0; i < m.Rows; i++ {
		mRow := m.SliceRow(i)
		xRow := x.SliceRow(i)
		r[i] = vec.CrossEntropyError(mRow, xRow)
	}
	return vec.Sum(r) / float64(m.Rows)
}

func (m *Matrix) equal(x *Matrix) bool {
	if m.Rows == x.Rows &&
		m.Columns == x.Columns &&
		vec.Equal(m.Vector, x.Vector) {
		return true
	}
	return false
}

//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//

func matMat(a Arithmetic, m1, m2 *Matrix) *Matrix {
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
			return &Matrix{
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

		return &Matrix{
			Vector:  vector,
			Rows:    m1.Rows,
			Columns: m1.Columns,
		}
	}
	panic([]*Matrix{m1, m2})
}

func matVec(a Arithmetic, m1 *Matrix, m2 vec.Vector) *Matrix {
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
	return &Matrix{
		Vector:  vector,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}

func matFloat(a Arithmetic, m1 *Matrix, m2 float64) *Matrix {
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
	return &Matrix{
		Vector:  vector,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}

func vecMat(a Arithmetic, m1 vec.Vector, m2 *Matrix) *Matrix {
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
	return &Matrix{
		Vector:  vector,
		Rows:    m2.Rows,
		Columns: m2.Columns,
	}
}

func floatMat(a Arithmetic, m1 float64, m2 *Matrix) *Matrix {
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
	return &Matrix{
		Vector:  vector,
		Rows:    m2.Rows,
		Columns: m2.Columns,
	}
}
