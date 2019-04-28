package mat

import (
	"errors"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/num"
)

type Mat64 struct {
	Array   array.Array
	Rows    int
	Columns int
}

func (m *Mat64) Shape() (int, int) {
	if m.Rows == 1 {
		return m.Columns, 0
	}
	return m.Rows, m.Columns
}

func (m *Mat64) Element(r int, c int) float64 {
	index := r*m.Columns + c
	// log.Printf("Row: %d / Column: %d / Index: %d\n", r, c, index)
	return m.Array[index]
}

func NewMat64(row int, column int, array []float64) (*Mat64, error) {
	if row == 0 || column == 0 {
		return nil, errors.New("row/columns is zero.")
	}
	return &Mat64{
		Array:   array,
		Rows:    row,
		Columns: column,
	}, nil
}

func Equal(m1 *Mat64, m2 *Mat64) bool {
	if m1.Rows == m2.Rows &&
		m1.Columns == m2.Columns &&
		m1.Array.Equal(m2.Array) {
		return true
	}
	return false
}

func Mul(m1 *Mat64, m2 *Mat64) *Mat64 {
	// 左辺の行数と、右辺の列数があっていないの掛け算できない
	if m1.Columns != m2.Rows {
		return nil
	}
	mat := make([]float64, m1.Rows*m2.Columns)
	for i := 0; i < m1.Columns; i++ {
		for r := 0; r < m1.Rows; r++ {
			for c := 0; c < m2.Columns; c++ {
				index := r*m1.Columns + c
				mat[index] += m1.Element(r, i) * m2.Element(i, c)
			}
		}
	}
	return &Mat64{
		Array:   mat,
		Rows:    m1.Rows,
		Columns: m2.Columns,
	}
}

func Add(m1 *Mat64, m2 *Mat64) *Mat64 {
	// 左辺の行数と、右辺の列数があっていないの掛け算できない
	if m1.Columns != m2.Columns && m1.Rows != m2.Rows {
		return nil
	}

	mat := make([]float64, m1.Rows*m1.Columns)
	for r := 0; r < m1.Rows; r++ {
		for c := 0; c < m2.Columns; c++ {
			index := r*m1.Columns + c
			mat[index] = m1.Element(r, c) + m2.Element(r, c)
		}
	}
	return &Mat64{
		Array:   mat,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}

func MulScalar(a float64, m *Mat64) *Mat64 {
	// 左辺の行数と、右辺の列数があっていないの掛け算できない
	mat := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			index := r*m.Columns + c
			mat[index] = a * m.Element(r, c)
		}
	}
	return &Mat64{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Sigmoid(m *Mat64) *Mat64 {
	mat := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			index := r*m.Columns + c
			mat[index] = num.Sigmoid(m.Element(r, c))
		}
	}
	return &Mat64{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Relu(m *Mat64) *Mat64 {
	mat := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			index := r*m.Columns + c
			mat[index] = num.Relu(m.Element(r, c))
		}
	}
	return &Mat64{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}


