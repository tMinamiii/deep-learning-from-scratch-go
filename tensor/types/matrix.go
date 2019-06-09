package types

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/vec"
)

type Matrix struct {
	Vector  vec.Vector
	Rows    int
	Columns int
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
