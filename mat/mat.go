package mat

import (
	"errors"
	"fmt"
	"sync"

	"github.com/naronA/zero_deeplearning/vec"
)

type Matrix struct {
	Array   vec.Vector
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
	return m.Array[r*m.Columns+c]
}

func (m *Matrix) SliceRow(r int) vec.Vector {
	slice := m.Array[r*m.Columns : (r+1)*m.Columns]
	return slice
}

func (m *Matrix) SliceColumn(c int) vec.Vector {
	slice := vec.Zeros(m.Columns)
	for i := 0; i < m.Rows; i++ {
		slice[i] = m.Element(i, c)
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
	zeros := vec.Zeros(rows * cols)
	return &Matrix{
		Array:   zeros,
		Rows:    rows,
		Columns: cols,
	}
}

func ZerosLike(x *Matrix) *Matrix {
	zeros := vec.Zeros(x.Rows * x.Columns)
	return &Matrix{
		Array:   zeros,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func NewMatrix(row int, column int, vec vec.Vector) (*Matrix, error) {
	if row == 0 || column == 0 {
		return nil, errors.New("row/columns is zero")
	}
	return &Matrix{
		Array:   vec,
		Rows:    row,
		Columns: column,
	}, nil
}

func NewRandnMatrix(row int, column int) (*Matrix, error) {
	if row == 0 || column == 0 {
		return nil, errors.New("row/columns is zero")
	}
	vec := vec.Randn(row * column)
	return &Matrix{
		Array:   vec,
		Rows:    row,
		Columns: column,
	}, nil
}

func (m *Matrix) NotEqual(m2 *Matrix) bool {
	return !m.Equal(m2)
}

func (m *Matrix) Equal(m2 *Matrix) bool {
	if m.Rows == m2.Rows &&
		m.Columns == m2.Columns &&
		m.Array.Equal(m2.Array) {
		return true
	}
	return false
}

func (m *Matrix) DotGo(m2 *Matrix) *Matrix {
	if m.Columns != m2.Rows {
		return nil
	}
	sum := vec.Zeros(m.Rows * m2.Columns)
	wg := &sync.WaitGroup{}
	ch := make(chan int)
	for i := 0; i < m.Columns; i++ {
		wg.Add(1)
		go func(ch chan int) {
			defer wg.Done()
			i := <-ch
			for c := 0; c < m2.Columns; c++ {
				for r := 0; r < m.Rows; r++ {
					sum[r*m2.Columns+c] += m.Element(r, i) * m2.Element(i, c)
				}
			}
		}(ch)
		ch <- i
	}
	wg.Wait()
	close(ch)
	return &Matrix{
		Array:   sum,
		Rows:    m.Rows,
		Columns: m2.Columns,
	}
}

func Dot(m1 *Matrix, m2 *Matrix) *Matrix {
	if m1.Columns != m2.Rows {
		return nil
	}
	mat := vec.Zeros(m1.Rows * m2.Columns)
	for i := 0; i < m1.Columns; i++ {
		for c := 0; c < m2.Columns; c++ {
			for r := 0; r < m1.Rows; r++ {
				mat[r*m2.Columns+c] += m1.Element(r, i) * m2.Element(i, c)
			}
		}
	}
	return &Matrix{
		Array:   mat,
		Rows:    m1.Rows,
		Columns: m2.Columns,
	}
}

func isTheSameShape(m1 *Matrix, m2 *Matrix) bool {
	if m1.Columns == m2.Columns && m1.Rows == m2.Rows {
		return true
	}
	return false
}

func (m *Matrix) Add(arg interface{}) *Matrix {
	switch v := arg.(type) {
	case *Matrix:
		if !isTheSameShape(m, v) {
			return nil
		}
		mat := m.Array.Add(v.Array)
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	case vec.Vector:
		if m.Columns != len(v) {
			return nil
		}
		mat := make(vec.Vector, m.Rows*m.Columns)
		for r := 0; r < m.Rows; r++ {
			for c := 0; c < len(v); c++ {
				index := r*m.Columns + c
				mat[index] = m.Element(r, c) + v[c]
			}
		}
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	case float64:
		mat := m.Array.Add(v)
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	default:
		return nil
	}
}

func (m *Matrix) Sub(arg interface{}) *Matrix {
	switch v := arg.(type) {
	case *Matrix:
		if !isTheSameShape(m, v) {
			return nil
		}
		mat := m.Array.Sub(v.Array)
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	case vec.Vector:
		if m.Columns != len(v) {
			return nil
		}
		mat := make(vec.Vector, m.Rows*m.Columns)
		for r := 0; r < m.Rows; r++ {
			for c := 0; c < len(v); c++ {
				index := r*m.Columns + c
				mat[index] = m.Element(r, c) - v[c]
			}
		}
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	case float64:
		mat := m.Array.Sub(v)
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	default:
		return nil
	}
}

func (m *Matrix) Mul(arg interface{}) *Matrix {
	switch v := arg.(type) {
	case *Matrix:
		if !isTheSameShape(m, v) {
			return nil
		}
		mat := m.Array.Mul(v.Array)
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	case vec.Vector:
		if m.Columns != len(v) {
			return nil
		}
		mat := make(vec.Vector, m.Rows*m.Columns)
		for r := 0; r < m.Rows; r++ {
			for c := 0; c < len(v); c++ {
				index := r*m.Columns + c
				mat[index] = m.Element(r, c) * v[c]
			}
		}
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	case float64:
		mat := m.Array.Mul(v)
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	default:
		return nil
	}
}

func (m *Matrix) Div(arg interface{}) *Matrix {
	switch v := arg.(type) {
	case *Matrix:
		if !isTheSameShape(m, v) {
			return nil
		}
		mat := m.Array.Div(v.Array)
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	case vec.Vector:
		if m.Columns != len(v) {
			return nil
		}
		mat := make(vec.Vector, m.Rows*m.Columns)
		for r := 0; r < m.Rows; r++ {
			for c := 0; c < len(v); c++ {
				index := r*m.Columns + c
				mat[index] = m.Element(r, c) / v[c]
			}
		}
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	case float64:
		mat := m.Array.Div(v)
		return &Matrix{
			Array:   mat,
			Rows:    m.Rows,
			Columns: m.Columns,
		}
	default:
		return nil
	}
}

func Sigmoid(m *Matrix) *Matrix {
	mat := vec.Sigmoid(m.Array)
	return &Matrix{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Relu(m *Matrix) *Matrix {
	mat := vec.Relu(m.Array)
	return &Matrix{
		Array:   mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Log(m *Matrix) *Matrix {
	log := vec.Log(m.Array)
	return &Matrix{
		Array:   log,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func SumAll(m *Matrix) float64 {
	return vec.Sum(m.Array)
}

func Sum(m *Matrix, axis int) *Matrix {
	if axis == 0 {
		v := make(vec.Vector, m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Sum(col)
		}
		return &Matrix{
			Array:   v,
			Rows:    1,
			Columns: m.Rows,
		}
	} else if axis == 1 {
		v := make(vec.Vector, m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.SliceRow(i)
			v[i] = vec.Sum(row)
		}
		return &Matrix{
			Array:   v,
			Rows:    1,
			Columns: m.Rows,
		}
	}
	return nil
}

func ArgMaxAll(x *Matrix) int {
	return vec.ArgMax(x.Array)
}

func ArgMax(m *Matrix, axis int) []int {
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
	return nil
}

func Softmax(x *Matrix) *Matrix {
	softmax := vec.Softmax(x.Array)
	return &Matrix{
		Array:   softmax,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func CrossEntropyError(y, t *Matrix) float64 {
	r := make(vec.Vector, y.Rows)
	for i := 0; i < y.Rows; i++ {
		yRow := y.SliceRow(i)
		tRow := t.SliceRow(i)
		r[i] = vec.CrossEntropyError(yRow, tRow)
	}
	return vec.Sum(r) / float64(y.Rows)
}

func NumericalGradient(f func(vec.Vector) float64, x *Matrix) *Matrix {
	grad := vec.NumericalGradient(f, x.Array)
	mat := &Matrix{Rows: x.Rows, Columns: x.Columns, Array: grad}
	return mat
}
