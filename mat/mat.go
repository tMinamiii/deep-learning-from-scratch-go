package mat

import (
	"errors"
	"fmt"
	"sync"

	"github.com/naronA/zero_deeplearning/vec"
)

type Matrix struct {
	Vector  vec.Vector
	Rows    int
	Columns int
}

func (m *Matrix) T() *Matrix {
	trans := vec.Vector{}
	for i := 0; i < m.Columns; i++ {
		col := m.SliceColumn(i)
		trans = append(trans, col...)
	}
	return &Matrix{
		Vector:  trans,
		Rows:    m.Columns,
		Columns: m.Rows,
	}
}

func (m *Matrix) Shape() (int, int) {
	if m.Rows == 1 {
		return m.Columns, 0
	}
	return m.Rows, m.Columns
}

func (m *Matrix) Element(r int, c int) float64 {
	return m.Vector[r*m.Columns+c]
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
		Vector:  zeros,
		Rows:    rows,
		Columns: cols,
	}
}

func ZerosLike(x *Matrix) *Matrix {
	zeros := vec.Zeros(x.Rows * x.Columns)
	return &Matrix{
		Vector:  zeros,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func NewMatrix(row int, column int, vec vec.Vector) (*Matrix, error) {
	if row == 0 || column == 0 {
		return nil, errors.New("row/columns is zero")
	}
	return &Matrix{
		Vector:  vec,
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
		Vector:  vec,
		Rows:    row,
		Columns: column,
	}, nil
}

func NotEqual(m1 *Matrix, m2 *Matrix) bool {
	return !Equal(m1, m2)
}

func Equal(m1 *Matrix, m2 *Matrix) bool {
	if m1.Rows == m2.Rows &&
		m1.Columns == m2.Columns &&
		vec.Equal(m1.Vector, m2.Vector) {
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
		Vector:  sum,
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
		Vector:  mat,
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

type Arithmetic int

const (
	ADD Arithmetic = iota
	SUB
	MUL
	DIV
)

func matMat(a Arithmetic, m1, m2 *Matrix) *Matrix {
	if !isTheSameShape(m1, m2) {
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
	return nil
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

func Add(x1 interface{}, x2 interface{}) *Matrix {
	if mat, ok := x1.(*Matrix); ok {
		switch x2v := x2.(type) {
		case *Matrix:
			return matMat(ADD, mat, x2v)
		case vec.Vector:
			return matVec(ADD, mat, x2v)
		case float64:
			return matFloat(ADD, mat, x2v)
		}
	} else if mat, ok := x2.(*Matrix); ok {
		switch x1v := x1.(type) {
		case vec.Vector:
			return matVec(ADD, mat, x1v)
		case float64:
			return matFloat(ADD, mat, x1v)
		}
	}
	return nil
}
func Sub(x1 interface{}, x2 interface{}) *Matrix {
	if mat, ok := x1.(*Matrix); ok {
		switch x2v := x2.(type) {
		case *Matrix:
			return matMat(SUB, mat, x2v)
		case vec.Vector:
			return matVec(SUB, mat, x2v)
		case float64:
			return matFloat(SUB, mat, x2v)
		}
	} else if mat, ok := x2.(*Matrix); ok {
		switch x1v := x1.(type) {
		case vec.Vector:
			return matVec(SUB, mat, x1v)
		case float64:
			return matFloat(SUB, mat, x1v)
		}
	}
	return nil
}

func Mul(x1 interface{}, x2 interface{}) *Matrix {
	if mat, ok := x1.(*Matrix); ok {
		switch x2v := x2.(type) {
		case *Matrix:
			return matMat(MUL, mat, x2v)
		case vec.Vector:
			return matVec(MUL, mat, x2v)
		case float64:
			return matFloat(MUL, mat, x2v)
		}
	} else if mat, ok := x2.(*Matrix); ok {
		switch x1v := x1.(type) {
		case vec.Vector:
			return matVec(MUL, mat, x1v)
		case float64:
			return matFloat(MUL, mat, x1v)
		}
	}
	return nil
}

func Div(x1 interface{}, x2 interface{}) *Matrix {
	if mat, ok := x1.(*Matrix); ok {
		switch x2v := x2.(type) {
		case *Matrix:
			return matMat(DIV, mat, x2v)
		case vec.Vector:
			return matVec(DIV, mat, x2v)
		case float64:
			return matFloat(DIV, mat, x2v)
		}
	} else if mat, ok := x2.(*Matrix); ok {
		switch x1v := x1.(type) {
		case vec.Vector:
			return matVec(DIV, mat, x1v)
		case float64:
			return matFloat(DIV, mat, x1v)
		}
	}
	return nil
}

func Sigmoid(m *Matrix) *Matrix {
	mat := vec.Sigmoid(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Relu(m *Matrix) *Matrix {
	mat := vec.Relu(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func Log(m *Matrix) *Matrix {
	log := vec.Log(m.Vector)
	return &Matrix{
		Vector:  log,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func SumAll(m *Matrix) float64 {
	return vec.Sum(m.Vector)
}

func Sum(m *Matrix, axis int) *Matrix {
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
	return nil
}

func MaxAll(x *Matrix) float64 {
	return vec.Max(x.Vector)
}

func Max(m *Matrix, axis int) vec.Vector {
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
	return nil
}

func ArgMaxAll(x *Matrix) int {
	return vec.ArgMax(x.Vector)
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

func Pow(x *Matrix, p float64) *Matrix {
	sqrt := vec.Pow(x.Vector, p)
	return &Matrix{
		Vector:  sqrt,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func Sqrt(x *Matrix) *Matrix {
	sqrt := vec.Sqrt(x.Vector)
	return &Matrix{
		Vector:  sqrt,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func Abs(x *Matrix) *Matrix {
	abs := vec.Abs(x.Vector)
	return &Matrix{
		Vector:  abs,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

func Exp(x *Matrix) *Matrix {
	exp := vec.Exp(x.Vector)
	return &Matrix{
		Vector:  exp,
		Rows:    x.Rows,
		Columns: x.Columns,
	}
}

/*
x = x.T
x = x - np.max(x, axis=0)
y = np.exp(x) / np.sum(np.exp(x), axis=0)
return y.T
*/
func Softmax(x *Matrix) *Matrix {
	xt := x.T()
	sub := Sub(xt, Max(xt, 0))
	expX := Exp(sub)
	sumExpX := Sum(expX, 0)
	softmax := Div(expX, sumExpX.Vector)
	return softmax.T()
}

/*
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

*/
func CrossEntropyError(y, t *Matrix) float64 {
	r := vec.Zeros(y.Rows)
	for i := 0; i < y.Rows; i++ {
		yRow := y.SliceRow(i)
		tRow := t.SliceRow(i)
		r[i] = vec.CrossEntropyError(yRow, tRow)
	}
	return vec.Sum(r) / float64(y.Rows)
}

func NumericalGradient(f func(vec.Vector) float64, x *Matrix) *Matrix {
	grad := vec.NumericalGradient(f, x.Vector)
	mat := &Matrix{Rows: x.Rows, Columns: x.Columns, Vector: grad}
	return mat
}
