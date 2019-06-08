package num

import (
	"errors"
	"fmt"

	"github.com/naronA/zero_deeplearning/vec"
)

type Matrix struct {
	Vector  vec.Vector
	Rows    int
	Columns int
}

func (m *Matrix) RowVecs() []vec.Vector {
	v := make([]vec.Vector, 0, m.Rows)
	for i := 0; i < m.Rows; i++ {
		col := m.SliceRow(i)
		v = append(v, col)
	}
	return v
}

func (m *Matrix) ColumnVecs() []vec.Vector {
	v := make([]vec.Vector, 0, m.Columns)
	for i := 0; i < m.Columns; i++ {
		col := m.SliceColumn(i)
		v = append(v, col)
	}
	return v
}

func (m *Matrix) T() *Matrix {
	return m.Transpose(1, 0)
}

func (m *Matrix) Transpose(a, b int) *Matrix {
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

func (m *Matrix) Window(x, y, h, w int) *Matrix {
	mat := Zeros(h, w)
	for i := x; i < x+h; i++ {
		for j := y; j < y+w; j++ {
			mat.Vector[(i-x)*w+(j-y)] = m.Element(i, j)
		}
	}
	return mat
}

func (m *Matrix) AssignWindow(window *Matrix, x, y, h, w int) {
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			val := window.Element(i, j)
			m.Assign(val, i+x, j+y)
		}
	}
}

func (m *Matrix) Pad(pad int) *Matrix {
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

func (m *Matrix) Reshape(row, col int) *Matrix {
	r, c := m.Shape()
	size := r * c
	if row == -1 {
		row = size / col
	} else if col == -1 {
		col = size / row
	}
	if m.Rows*m.Columns != row*col {
		return nil
	}

	return &Matrix{
		Vector:  m.Vector,
		Rows:    row,
		Columns: col,
	}
}

func (m *Matrix) ReshapeTo4D(a, b, c, d int) Tensor4D {
	row, col := m.Shape()
	size := row * col

	if a == -1 {
		a = int(size / b / c / d)
	} else if b == -1 {
		b = int(size / a / c / d)
	} else if c == -1 {
		c = int(size / a / b / d)
	} else if d == -1 {
		d = int(size / a / b / c)
	}

	t4d := ZerosT4D(a, b, c, d)
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			sv := m.Vector[(i*b+j)*c*d : (i*b+j+1)*c*d]
			t4d[i][j] = &Matrix{
				Vector:  sv,
				Rows:    c,
				Columns: d,
			}
		}
	}
	return t4d
}

func (m *Matrix) ReshapeTo5D(a, b, c, d, e int) Tensor5D {
	row, col := m.Shape()
	size := row * col

	if a == -1 {
		a = int(size / b / c / d / e)
	} else if b == -1 {
		b = int(size / a / c / d / e)
	} else if c == -1 {
		c = int(size / a / b / d / e)
	} else if d == -1 {
		d = int(size / a / b / c / e)
	} else if e == -1 {
		e = int(size / a / b / c / d)
	}

	t5d := ZerosT5D(a, b, c, d, e)
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			for k := 0; k < c; k++ {
				sv := m.Vector[((i*b+j)*c+k)*d*e : ((i*b+j)*c+k+1)*d*e]
				t5d[i][j][k] = &Matrix{
					Vector:  sv,
					Rows:    c,
					Columns: d,
				}
			}
		}
	}
	return t5d
}

func (m *Matrix) ReshapeTo6D(a, b, c, d, e, f int) Tensor6D {
	t6d := ZerosT6D(a, b, c, d, e, f)
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			for k := 0; k < c; k++ {
				for l := 0; l < d; l++ {
					sv := m.Vector[(((i*b+j)*c+k)*d+l)*e*f : (((i*b+j)*c+k)*d+l+1)*e*f]
					t6d[i][j][k][l] = &Matrix{
						Vector:  sv,
						Rows:    e,
						Columns: f,
					}
				}
			}
		}
	}
	return t6d
}

func (m *Matrix) Element(r int, c int) float64 {
	return m.Vector[r*m.Columns+c]
}

func (m *Matrix) Assign(value float64, r, c int) {
	m.Vector[r*m.Columns+c] = value
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

func (m *Matrix) Col2Img(shape []int, fh, fw, stride, pad int) Tensor4D {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	outH := (H+2*pad-fh)/stride + 1
	outW := (W+2*pad-fw)/stride + 1
	ncol := m.ReshapeTo6D(N, outH, outW, C, fh, fw).Transpose(0, 3, 4, 5, 1, 2)
	img := ZerosT4D(N, C, H+2*pad+stride-1, W+2*pad+stride-1)
	// imgMats := make(Tensor3D{})
	// for _, imgT3D := range img {
	// 	imgMats = append(imgMats, imgT3D...)
	// }
	for y := 0; y < fh; y++ {
		yMax := y + stride*outH
		for x := 0; x < fw; x++ {
			xMax := x + stride*outW
			slice := img.StrideSlice(y, yMax, x, xMax, stride)
			ncolSlice := ncol.Slice(y, x)
			AddAssign(slice, ncolSlice)
		}
	}
	return img.Slice(pad, H+pad, pad, W+pad)
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

func NewRandnMatrix(row, column int) (*Matrix, error) {
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

func NotEqual(m1, m2 *Matrix) bool {
	return !Equal(m1, m2)
}

func Equal(m1, m2 *Matrix) bool {
	if m1.Rows == m2.Rows &&
		m1.Columns == m2.Columns &&
		vec.Equal(m1.Vector, m2.Vector) {
		return true
	}
	return false
}

func dotPart(i int, a, b, c *Matrix, ch chan int) {
	ac := a.Columns
	bc := b.Columns
	for j := 0; j < bc; j++ {
		part := 0.0
		for k := 0; k < ac; k++ {
			part += a.Vector[i*a.Columns+k] * b.Vector[k*b.Columns+j]
			//part += a.Element(i, k) * b.Element(k, j)
		}
		c.Vector[i*c.Columns+j] = part
		// c.Assign(part, i, j)
	}
	ch <- i
}

func Dot(m1, m2 *Matrix) *Matrix {
	if m1.Columns != m2.Rows {
		return nil
	}
	// fmt.Println(m1.Rows * m2.Columns)
	v3 := vec.Zeros(m1.Rows * m2.Columns)
	m3 := &Matrix{
		Vector:  v3,
		Rows:    m1.Rows,
		Columns: m2.Columns,
	}

	ch := make(chan int)
	for i := 0; i < m1.Rows; i++ {
		go dotPart(i, m1, m2, m3, ch)
	}
	for i := 0; i < m1.Rows; i++ {
		<-ch
	}
	return m3
}

func Dot2(m1, m2 *Matrix) *Matrix {
	if m1.Columns != m2.Rows {
		return nil
	}
	// fmt.Println(m1.Rows * m2.Columns)
	mat := vec.Zeros(m1.Rows * m2.Columns)
	for i := 0; i < m1.Columns; i++ {
		for c := 0; c < m2.Columns; c++ {
			for r := 0; r < m1.Rows; r++ {
				// mat[r*m2.Columns+c] += m1.Element(r, i) * m2.Element(i, c)
				mat[r*m2.Columns+c] += m1.Vector[r*m1.Columns+i] * m2.Vector[i*m2.Columns+c]
			}
		}
	}
	return &Matrix{
		Vector:  mat,
		Rows:    m1.Rows,
		Columns: m2.Columns,
	}
}

func isTheSameShape(m1, m2 *Matrix) bool {
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

func Add(x1, x2 interface{}) *Matrix {
	if x1v, ok := x1.(*Matrix); ok {
		switch x2v := x2.(type) {
		case *Matrix:
			return matMat(ADD, x1v, x2v)
		case vec.Vector:
			return matVec(ADD, x1v, x2v)
		case float64:
			return matFloat(ADD, x1v, x2v)
		case int:
			return matFloat(ADD, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(*Matrix); ok {
		switch x1v := x1.(type) {
		case vec.Vector:
			return vecMat(ADD, x1v, x2v)
		case float64:
			return floatMat(ADD, x1v, x2v)
		case int:
			return floatMat(ADD, float64(x1v), x2v)
		}
	}
	return nil
}
func Sub(x1, x2 interface{}) *Matrix {
	if x1v, ok := x1.(*Matrix); ok {
		switch x2v := x2.(type) {
		case *Matrix:
			return matMat(SUB, x1v, x2v)
		case vec.Vector:
			return matVec(SUB, x1v, x2v)
		case float64:
			return matFloat(SUB, x1v, x2v)
		case int:
			return matFloat(SUB, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(*Matrix); ok {
		switch x1v := x1.(type) {
		case vec.Vector:
			return vecMat(SUB, x1v, x2v)
		case float64:
			return floatMat(SUB, x1v, x2v)
		case int:
			return floatMat(SUB, float64(x1v), x2v)

		}
	}
	return nil
}

func Mul(x1, x2 interface{}) *Matrix {
	if x1v, ok := x1.(*Matrix); ok {
		switch x2v := x2.(type) {
		case *Matrix:
			return matMat(MUL, x1v, x2v)
		case vec.Vector:
			return matVec(MUL, x1v, x2v)
		case float64:
			return matFloat(MUL, x1v, x2v)
		case int:
			return matFloat(MUL, x1v, float64(x2v))

		}
	} else if x2v, ok := x2.(*Matrix); ok {
		switch x1v := x1.(type) {
		case vec.Vector:
			return vecMat(MUL, x1v, x2v)
		case float64:
			return floatMat(MUL, x1v, x2v)
		case int:
			return floatMat(MUL, float64(x1v), x2v)
		}
	}
	return nil
}

func Div(x1, x2 interface{}) *Matrix {
	if x1v, ok := x1.(*Matrix); ok {
		switch x2v := x2.(type) {
		case *Matrix:
			return matMat(DIV, x1v, x2v)
		case vec.Vector:
			return matVec(DIV, x1v, x2v)
		case float64:
			return matFloat(DIV, x1v, x2v)
		case int:
			return matFloat(DIV, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(*Matrix); ok {
		switch x1v := x1.(type) {
		case vec.Vector:
			return vecMat(DIV, x1v, x2v)
		case float64:
			return floatMat(DIV, x1v, x2v)
		case int:
			return floatMat(DIV, float64(x1v), x2v)
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
func MeanAll(m *Matrix) float64 {
	return vec.Sum(m.Vector) / float64(len(m.Vector))
}
func SumAll(m *Matrix) float64 {
	return vec.Sum(m.Vector)
}

func Mean(m *Matrix, axis int) *Matrix {
	if axis == 0 {
		v := vec.Zeros(m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Sum(col) / float64(m.Rows)
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
			v[i] = vec.Sum(row) / float64(m.Columns)
		}
		return &Matrix{
			Vector:  v,
			Rows:    1,
			Columns: m.Rows,
		}
	}
	return nil
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

func Softmax(x *Matrix) *Matrix {
	xt := x.T()
	sub := Sub(xt, Max(xt, 0))
	expX := Exp(sub)
	sumExpX := Sum(expX, 0)
	softmax := Div(expX, sumExpX.Vector)
	return softmax.T()
}

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
