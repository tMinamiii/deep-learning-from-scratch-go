package mat

import (
	"errors"
	"fmt"
	"log"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/scalar"
)

type Matrix struct {
	Array   array.Array
	Rows    int
	Columns int
}

func (m *Matrix) Shape() (int, int) {
	if m.Rows == 1 {
		return m.Columns, 0
	}
	return m.Rows, m.Columns
}

func (m *Matrix) Element(r int, c int) float64 {
	index := r*m.Columns + c
	if len(m.Array) <= index {
		log.Println(len(m.Array), index)
		log.Printf("Row: %d / Column: %d / Index: %d\n", r, c, index)
	}
	return m.Array[index]
}

func (m *Matrix) SliceRow(r int) array.Array {
	slice := make(array.Array, m.Columns)
	for i := 0; i < len(slice); i++ {
		slice[i] = m.Array[i+r*m.Columns]
	}
	return slice
}

func (m *Matrix) String() string {
	str := "[\n"
	for i := 0; i < m.Rows; i++ {
		str += fmt.Sprintf("  %v,\n", m.SliceRow(i))
	}
	str += "]"
	return str
}

func Zeros(rows int, cols int) *Matrix {
	zeros := make(array.Array, rows*cols)
	for i := range zeros {
		zeros[i] = 0
	}
	mat, err := NewMat64(rows, cols, zeros)
	if err != nil {
		panic(err)
	}
	return mat
}

func ZerosLike(x *Matrix) *Matrix {
	zeros := make(array.Array, x.Rows*x.Columns)
	for i := range zeros {
		zeros[i] = 0
	}
	mat, err := NewMat64(x.Rows, x.Columns, zeros)
	if err != nil {
		panic(err)
	}
	return mat
}

func NewMat64(row int, column int, array array.Array) (*Matrix, error) {
	if row == 0 || column == 0 {
		return nil, errors.New("row/columns is zero.")
	}
	return &Matrix{
		Array:   array,
		Rows:    row,
		Columns: column,
	}, nil
}

func NewRandnMat64(row int, column int) (*Matrix, error) {
	if row == 0 || column == 0 {
		return nil, errors.New("row/columns is zero.")
	}
	array := array.Randn(row * column)
	return &Matrix{
		Array:   array,
		Rows:    row,
		Columns: column,
	}, nil
}

func (m1 *Matrix) NotEqual(m2 *Matrix) bool {
	if m1.Rows == m2.Rows &&
		m1.Columns == m2.Columns &&
		m1.Array.Equal(m2.Array) {
		return false
	}
	return true
}

func (m1 *Matrix) Equal(m2 *Matrix) bool {
	if m1.Rows == m2.Rows &&
		m1.Columns == m2.Columns &&
		m1.Array.Equal(m2.Array) {
		return true
	}
	return false
}

func (m1 *Matrix) Dot(m2 *Matrix) *Matrix {
	if m1.Columns != m2.Rows {
		return nil
	}
	mat := make([]float64, m1.Rows*m2.Columns)
	for r := 0; r < m1.Rows; r++ {
		for c := 0; c < m2.Columns; c++ {
			for i := 0; i < m1.Columns; i++ {
				index := r*m2.Columns + c
				mat[index] += m1.Element(r, i) * m2.Element(i, c)
			}
		}
	}
	return &Matrix{
		Array:   mat,
		Rows:    m1.Rows,
		Columns: m2.Columns,
	}
}

func (m1 *Matrix) Mul(m2 *Matrix) *Matrix {
	mul := m1.Array.Multi(m2.Array)
	return &Matrix{
		Array:   mul,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}

func (m1 *Matrix) Add(m2 *Matrix) *Matrix {
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
	return &Matrix{
		Array:   mat,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}

func (m1 *Matrix) AddBroadCast(m2 *Matrix) *Matrix {
	// 左辺の行数と、右辺の列数があっていないの掛け算できない
	if m1.Columns != m2.Columns {
		return nil
	}

	mat := make([]float64, m1.Rows*m1.Columns)
	for r := 0; r < m1.Rows; r++ {
		for c := 0; c < m1.Columns; c++ {
			index := r*m1.Columns + c
			mat[index] = m1.Element(r, c) + m2.Element(0, c)
		}
	}
	return &Matrix{
		Array:   mat,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}
func (m *Matrix) AddAll(a float64) *Matrix {

	mat := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			index := r*m.Columns + c
			mat[index] = m.Element(r, c) + a
		}
	}
	return &Matrix{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Matrix) MulAll(a float64) *Matrix {
	// 左辺の行数と、右辺の列数があっていないの掛け算できない
	mat := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			index := r*m.Columns + c
			mat[index] = a * m.Element(r, c)
		}
	}
	return &Matrix{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Sigmoid(m *Matrix) *Matrix {
	mat := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			index := r*m.Columns + c
			mat[index] = scalar.Sigmoid(m.Element(r, c))
		}
	}
	return &Matrix{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Relu(m *Matrix) *Matrix {
	mat := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			index := r*m.Columns + c
			mat[index] = scalar.Relu(m.Element(r, c))
		}
	}
	return &Matrix{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Log(m *Matrix) *Matrix {
	log := array.Log(m.Array)
	return &Matrix{
		Array:   log,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Sum(m *Matrix) float64 {
	return array.Sum(m.Array)
}

func ArgMax(x *Matrix) []int {
	r := make([]int, x.Rows)
	for i := 0; i < x.Rows; i++ {
		row := x.SliceRow(i)
		r[i] = array.ArgMax(row)
	}
	return r
}

func Softmax(x *Matrix) *Matrix {
	m := array.Array{}
	for i := 0; i < x.Rows; i++ {
		xRow := x.SliceRow(i)
		m = append(m, array.Softmax(xRow)...)
	}
	r, err := NewMat64(x.Rows, x.Columns, m)
	if err != nil {
		panic(err)
	}
	return r
}

func CrossEntropyError(y, t *Matrix) float64 {
	r := make(array.Array, y.Rows)
	for i := 0; i < y.Rows; i++ {
		yRow := y.SliceRow(i)
		tRow := t.SliceRow(i)
		r[i] = array.CrossEntropyError(yRow, tRow)
	}
	return array.Sum(r) / float64(y.Rows)
}

func NumericalGradient(f func(array.Array) float64, x *Matrix) *Matrix {
	grad := array.NumericalGradient(f, x.Array)
	mat, _ := NewMat64(x.Rows, x.Columns, grad)
	return mat
}
