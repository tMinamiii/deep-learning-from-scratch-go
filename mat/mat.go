package mat

import (
	"errors"
	"fmt"
	"log"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/scalar"
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
	if len(m.Array) <= index {
		log.Println(len(m.Array), index)
		log.Printf("Row: %d / Column: %d / Index: %d\n", r, c, index)
	}
	return m.Array[index]
}

func (m *Mat64) SliceRow(r int) array.Array {
	slice := make(array.Array, m.Columns)
	for i := 0; i < len(slice); i++ {
		slice[i] = m.Array[i+r*m.Columns]
	}
	return slice
}

func (m *Mat64) String() string {
	str := "[\n"
	for i := 0; i < m.Rows; i++ {
		str += fmt.Sprintf("  %v,\n", m.SliceRow(i))
	}
	str += "]"
	return str
}

func Zeros(rows int, cols int) *Mat64 {
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

func ZerosLike(x *Mat64) *Mat64 {
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

func NewMat64(row int, column int, array array.Array) (*Mat64, error) {
	if row == 0 || column == 0 {
		return nil, errors.New("row/columns is zero.")
	}
	return &Mat64{
		Array:   array,
		Rows:    row,
		Columns: column,
	}, nil
}

func NewRandnMat64(row int, column int) (*Mat64, error) {
	if row == 0 || column == 0 {
		return nil, errors.New("row/columns is zero.")
	}
	array := array.Randn(row * column)
	return &Mat64{
		Array:   array,
		Rows:    row,
		Columns: column,
	}, nil
}

func (m1 *Mat64) NotEqual(m2 *Mat64) bool {
	if m1.Rows == m2.Rows &&
		m1.Columns == m2.Columns &&
		m1.Array.Equal(m2.Array) {
		return false
	}
	return true
}

func (m1 *Mat64) Equal(m2 *Mat64) bool {
	if m1.Rows == m2.Rows &&
		m1.Columns == m2.Columns &&
		m1.Array.Equal(m2.Array) {
		return true
	}
	return false
}

func (m1 *Mat64) Dot(m2 *Mat64) *Mat64 {
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
	return &Mat64{
		Array:   mat,
		Rows:    m1.Rows,
		Columns: m2.Columns,
	}
}

func (m1 *Mat64) Mul(m2 *Mat64) *Mat64 {
	mul := m1.Array.Multi(m2.Array)
	return &Mat64{
		Array:   mul,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}

func (m1 *Mat64) Add(m2 *Mat64) *Mat64 {
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

func (m1 *Mat64) AddBroadCast(m2 *Mat64) *Mat64 {
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
	return &Mat64{
		Array:   mat,
		Rows:    m1.Rows,
		Columns: m1.Columns,
	}
}
func (m *Mat64) AddAll(a float64) *Mat64 {

	mat := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			index := r*m.Columns + c
			mat[index] = m.Element(r, c) + a
		}
	}
	return &Mat64{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Mat64) MulAll(a float64) *Mat64 {
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
			mat[index] = scalar.Sigmoid(m.Element(r, c))
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
			mat[index] = scalar.Relu(m.Element(r, c))
		}
	}
	return &Mat64{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Log(m *Mat64) *Mat64 {
	log := array.Log(m.Array)
	return &Mat64{
		Array:   log,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Sum(m *Mat64) float64 {
	return array.Sum(m.Array)
}

func ArgMax(x *Mat64) []int {
	r := make([]int, x.Rows)
	for i := 0; i < x.Rows; i++ {
		row := x.SliceRow(i)
		r[i] = array.ArgMax(row)
	}
	return r
}

func Softmax(x *Mat64) *Mat64 {
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

func CrossEntropyError(y, t *Mat64) float64 {
	r := make(array.Array, y.Rows)
	for i := 0; i < y.Rows; i++ {
		yRow := y.SliceRow(i)
		tRow := t.SliceRow(i)
		r[i] = array.CrossEntropyError(yRow, tRow)
	}
	return array.Sum(r) / float64(y.Rows)
}

func NumericalGradient(f func(array.Array) float64, x *Mat64) *Mat64 {
	grad := array.NumericalGradient(f, x.Array)
	mat, _ := NewMat64(x.Rows, x.Columns, grad)
	return mat
}
