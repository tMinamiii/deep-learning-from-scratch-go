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

func (m *Matrix) t() *Matrix {
	return m.transpose(1, 0)
}

func (m *Matrix) Shape() (int, int) {
	if m.Rows == 1 {
		return m.Columns, -1
	}
	return m.Rows, m.Columns
}

// Matrixなのでndimは常に2
func (m *Matrix) ndim() int {
	return 2
}

func (m *Matrix) element(r int, c int) float64 {
	return m.Vector[r*m.Columns+c]
}

func (m *Matrix) assign(value float64, r, c int) {
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

func zerosLikeMat(m *Matrix) *Matrix {
	return zerosMat(m.Rows, m.Columns)
}

func (m *Matrix) window(x, y, h, w int) *Matrix {
	mat := zerosMat(h, w)
	for i := x; i < x+h; i++ {
		for j := y; j < y+w; j++ {
			mat.Vector[(i-x)*w+(j-y)] = m.element(i, j)
		}
	}
	return mat
}

func (m *Matrix) transpose(a, b int) *Matrix {
	trans := make(vec.Vector, m.Rows*m.Columns)
	if a == 0 && b == 1 {
		for i := 0; i < m.Rows; i++ {
			col := m.sliceRow(i)
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
		col := m.sliceColumn(i)
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

// func (m *Matrix) element(point []int) float64 {
// 	r := point[0]
// 	c := point[1]
// 	return m.Vector[r*m.Columns+c]
// }
//
// func (m *Matrix) assign(value float64, point []int) {
// 	a := point[0]
// 	b := point[1]
// 	m.Vector[a*m.Columns+b] = value
// }

func (m *Matrix) assignWindow(window *Matrix, x, y, h, w int) {
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			val := window.element(i, j)
			m.assign(val, i+x, j+y)
		}
	}
}

func zerosMat(rows, cols int) *Matrix {
	zeros := vec.Zeros(rows * cols)
	return &Matrix{
		Vector:  zeros,
		Rows:    rows,
		Columns: cols,
	}
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
		srow := m.sliceRow(j)
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
			col := m.sliceColumn(i)
			v[i] = vec.Max(col)
		}
		return v
	} else if axis == 1 {
		v := vec.Zeros(m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.sliceRow(i)
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
			col := m.sliceColumn(i)
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
			row := m.sliceRow(i)
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
			col := m.sliceColumn(i)
			v[i] = vec.ArgMax(col)
		}
		return v
	} else if axis == 1 {
		v := make([]int, m.Rows)
		for i := 0; i < m.Rows; i++ {
			row := m.sliceRow(i)
			v[i] = vec.ArgMax(row)
		}
		return v
	}
	panic(m)
}

func (m *Matrix) crossEntropyError(x *Matrix) float64 {
	r := vec.Zeros(m.Rows)
	for i := 0; i < m.Rows; i++ {
		mRow := m.sliceRow(i)
		xRow := x.sliceRow(i)
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

func matMat(a Arithmetic, m1, m2 *Matrix) *Matrix {
	if m1.Rows != m2.Rows && m1.Columns == m2.Columns {
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
						vector[index] = m1.element(r, c) + m2.element(0, c)
					case SUB:
						vector[index] = m1.element(r, c) - m2.element(0, c)
					case MUL:
						vector[index] = m1.element(r, c) * m2.element(0, c)
					case DIV:
						vector[index] = m1.element(r, c) / m2.element(0, c)
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
				vector[index] = m1.element(r, c) + m2[c]
			case SUB:
				vector[index] = m1.element(r, c) - m2[c]
			case MUL:
				vector[index] = m1.element(r, c) * m2[c]
			case DIV:
				vector[index] = m1.element(r, c) / m2[c]
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
				vector[index] = m1[c] + m2.element(r, c)
			case SUB:
				vector[index] = m1[c] - m2.element(r, c)
			case MUL:
				vector[index] = m1[c] * m2.element(r, c)
			case DIV:
				vector[index] = m1[c] / m2.element(r, c)
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

func (m *Matrix) exp() *Matrix {
	mat := vec.Exp(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Matrix) log() *Matrix {
	mat := vec.Log(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Matrix) maxAll() float64 {
	return vec.Max(m.Vector)
}

func (m *Matrix) meanAll() float64 {
	return vec.Sum(m.Vector) / float64(len(m.Vector))
}

func (m *Matrix) mean(axis int) *Matrix {
	if axis == 0 {
		v := vec.Zeros(m.Columns)
		for i := 0; i < m.Columns; i++ {
			col := m.sliceColumn(i)
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
			row := m.sliceRow(i)
			v[i] = vec.Sum(row) / float64(m.Columns)
		}
		return &Matrix{
			Vector:  v,
			Rows:    1,
			Columns: m.Rows,
		}
	}
	panic(m)
}

func (m *Matrix) pow(p float64) *Matrix {
	mat := vec.Pow(m.Vector, p)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Matrix) sumAll() float64 {
	return vec.Sum(m.Vector)
}

func (m *Matrix) sqrt() *Matrix {
	mat := vec.Sqrt(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Matrix) softmax() *Matrix {
	xt := m.t()
	sub := matVec(SUB, xt, xt.max(0))
	expX := sub.exp()
	sumExpX := expX.sum(0)
	softmax := matVec(DIV, expX, sumExpX.Vector)
	return softmax.t()
}

func (m *Matrix) sigmoid() *Matrix {
	mat := vec.Sigmoid(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Matrix) relu() *Matrix {
	mat := vec.Relu(m.Vector)
	return &Matrix{
		Vector:  mat,
		Rows:    m.Rows,
		Columns: m.Columns,
	}
}

func (m *Matrix) numericalGradient(f func(vec.Vector) float64) *Matrix {
	grad := vec.NumericalGradient(f, m.Vector)
	mat := &Matrix{Rows: m.Rows, Columns: m.Columns, Vector: grad}
	return mat
}

func (m *Matrix) isTheSameShapeMat(x *Matrix) bool {
	if m.Columns == x.Columns && m.Rows == x.Rows {
		return true
	}
	return false
}

func dotMatPart(i int, a, b, c *Matrix, ch chan int) {
	ac := a.Columns
	bc := b.Columns
	for j := 0; j < bc; j++ {
		part := 0.0
		for k := 0; k < ac; k++ {
			part += a.Vector[i*a.Columns+k] * b.Vector[k*b.Columns+j]
		}
		c.Vector[i*c.Columns+j] = part
	}
	ch <- i
}

func dotMat(m1, m2 *Matrix) *Matrix {
	if m1.Columns != m2.Rows {
		return nil
	}
	v3 := vec.Zeros(m1.Rows * m2.Columns)
	m3 := &Matrix{
		Vector:  v3,
		Rows:    m1.Rows,
		Columns: m2.Columns,
	}

	ch := make(chan int)
	for i := 0; i < m1.Rows; i++ {
		go dotMatPart(i, m1, m2, m3, ch)
	}
	for i := 0; i < m1.Rows; i++ {
		<-ch
	}
	return m3
}

func (m *Matrix) sliceRow(r int) vec.Vector {
	return m.Vector[r*m.Columns : (r+1)*m.Columns]
}

func (m *Matrix) sliceColumn(c int) vec.Vector {
	slice := vec.Zeros(m.Rows)
	for i := 0; i < m.Rows; i++ {
		slice[i] = m.element(i, c)
	}
	return slice
}

func (m *Matrix) reshape(row, col int) *Matrix {
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

func (m *Matrix) reshapeTo4D(a, b, c, d int) Tensor4D {
	row, col := m.Shape()
	size := row * col
	switch {
	case a == -1:
		a = size / b / c / d
	case b == -1:
		b = size / a / c / d
	case c == -1:
		c = size / a / b / d
	case d == -1:
		d = size / a / b / c
	}
	t4d := zerosT4D(a, b, c, d)
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

func (m *Matrix) reshapeTo5D(a, b, c, d, e int) Tensor5D {
	row, col := m.Shape()
	size := row * col
	switch {
	case a == -1:
		a = size / b / c / d / e
	case b == -1:
		b = size / a / c / d / e
	case c == -1:
		c = size / a / b / d / e
	case d == -1:
		d = size / a / b / c / e
	case e == -1:
		e = size / a / b / c / d
	}
	t5d := zerosT5D(a, b, c, d, e)
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

func (m *Matrix) reshapeTo6D(a, b, c, d, e, f int) Tensor6D {
	t6d := zerosT6D(a, b, c, d, e, f)
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

func (m *Matrix) col2Img(shape []int, fh, fw, stride, pad int) Tensor4D {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	outH := (H+2*pad-fh)/stride + 1
	outW := (W+2*pad-fw)/stride + 1
	ncol := m.reshapeTo6D(N, outH, outW, C, fh, fw).transpose(0, 3, 4, 5, 1, 2)
	// ncol := m.ReshapeTo6D(N, outH, outW, C, fh, fw).Transpose(0, 3, 4, 5, 1, 2)
	img := zerosT4D(N, C, H+2*pad+stride-1, W+2*pad+stride-1)
	// img := ZerosT4D(N, C, H+2*pad+stride-1, W+2*pad+stride-1)
	for y := 0; y < fh; y++ {
		yMax := y + stride*outH
		for x := 0; x < fw; x++ {
			xMax := x + stride*outW
			slice := img.strideSlice(y, yMax, x, xMax, stride)
			ncolSlice := ncol.sliceTo4D(y, x)
			addAssignT4D(slice, ncolSlice)
		}
	}
	return img.slice(pad, H+pad, pad, W+pad)
}
