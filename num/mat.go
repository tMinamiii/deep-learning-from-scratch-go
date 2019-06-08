package num

import (
	"fmt"
	"math"

	"github.com/naronA/zero_deeplearning/vec"
)

type Matrix []vec.Vector

func NewMatrix(v vec.Vector, row, col int) Matrix {
	totalLen := len(v)
	mat := Zeros(row, col)
	for i := 0; i < totalLen; i++ {
		r := i / col
		c := i % col
		mat[r][c] = v[i]
	}
	return mat
}

func (m Matrix) Assign(i int, value float64) {
	r := i / m.Columns()
	c := i % m.Columns()
	m[r][c] = value
}

func (m Matrix) Rows() int {
	return len(m)
}

func (m Matrix) Columns() int {
	return len(m[0])
}

func (m Matrix) Flatten() vec.Vector {
	flat := make(vec.Vector, 0, m.Rows()*m.Columns())
	for _, v := range m {
		flat = append(flat, v...)
	}
	return flat
}

func (m Matrix) RowVecs() []vec.Vector {
	v := make([]vec.Vector, 0, m.Rows())
	for i := 0; i < m.Rows(); i++ {
		v[i] = m[i]
	}
	return v
}

func (m Matrix) ColumnVecs() []vec.Vector {
	v := make([]vec.Vector, 0, m.Columns())
	for i := 0; i < m.Columns(); i++ {
		col := m.SliceColumn(i)
		v[i] = col
	}
	return v
}

func (m Matrix) T() Matrix {
	return m.Transpose(1, 0)
}

func (m Matrix) Transpose(a, b int) Matrix {
	if a == 0 && b == 1 {
		trans := Zeros(m.Rows(), m.Columns())
		for i := 0; i < m.Rows(); i++ {
			row := m.SliceRow(i)
			for j := 0; j < len(row); j++ {
				trans[i][j] = row[j]
			}
		}
		fmt.Println(trans)
		return trans
	}
	trans := Zeros(m.Columns(), m.Rows())
	for i := 0; i < m.Columns(); i++ {
		col := m.SliceColumn(i)
		for j, c := range col {
			trans[i][j] = c
		}
	}
	return trans
}

func (m Matrix) Window(x, y, h, w int) Matrix {
	mat := Zeros(h, w)
	for i := x; i < x+h; i++ {
		for j := y; j < y+w; j++ {
			mat[i-x][j-y] = m[i][j]
		}
	}
	return mat
}

func (m Matrix) AssignWindow(window Matrix, x, y, h, w int) {
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			val := window[i][j]
			m[i+x][j+y] = val
		}
	}
}

func (m Matrix) Pad(pad int) Matrix {
	if pad == 0 {
		return m
	}
	col := m.Columns()
	row := m.Rows()
	newVec := make([]vec.Vector, 0, row+2*pad)
	rowPad := vec.Zeros(col + 2*pad)
	for j := 0; j < pad; j++ {
		newVec = append(newVec, rowPad)
	}
	for j := 0; j < m.Rows(); j++ {
		row := make(vec.Vector, 0, (col + 2*pad))
		srow := m.SliceRow(j)
		for k := 0; k < pad; k++ {
			row = append(row, 0)
		}
		row = append(row, srow...)
		for k := 0; k < pad; k++ {
			row = append(row, 0)
		}
		newVec = append(newVec, row)
	}
	for j := 0; j < pad; j++ {
		newVec = append(newVec, rowPad)
	}
	return newVec
}

func (m Matrix) Shape() (int, int) {
	if m.Rows() == 1 {
		return m.Columns(), -1
	}
	return m.Rows(), m.Columns()
}

// Matrixなのでndimは常に2
func (m Matrix) Ndim() int {
	return 2
}

func (m Matrix) Reshape(row, col int) Matrix {
	r, c := m.Shape()
	size := r * c
	if row == -1 {
		row = size / col
	} else if col == -1 {
		col = size / row
	}
	if m.Rows()*m.Columns() != row*col {
		return nil
	}
	flat := m.Flatten()
	reshape := make([]vec.Vector, row)
	for i := range reshape {
		reshape[i] = flat[i*col : (i+1)*col]
	}

	return reshape
}

func (m Matrix) ReshapeTo4D(a, b, c, d int) Tensor4D {
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

	flat := m.Flatten()
	t4d := ZerosT4D(a, b, c, d)
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			v := flat[(i*b+j)*c*d : (i*b+j+1)*c*d]
			t4d[i][j] = NewMatrix(v, c, d)
		}
	}
	return t4d
}

func (m Matrix) ReshapeTo5D(a, b, c, d, e int) Tensor5D {
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

	flat := m.Flatten()
	t5d := ZerosT5D(a, b, c, d, e)
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			for k := 0; k < c; k++ {
				v := flat[((i*b+j)*c+k)*d*e : ((i*b+j)*c+k+1)*d*e]
				t5d[i][j][k] = NewMatrix(v, d, e)
			}
		}
	}
	return t5d
}

func (m Matrix) ReshapeTo6D(a, b, c, d, e, f int) Tensor6D {
	flat := m.Flatten()
	t6d := ZerosT6D(a, b, c, d, e, f)
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			for k := 0; k < c; k++ {
				for l := 0; l < d; l++ {
					v := flat[(((i*b+j)*c+k)*d+l)*e*f : (((i*b+j)*c+k)*d+l+1)*e*f]
					t6d[i][j][k][l] = NewMatrix(v, e, f)
				}
			}
		}
	}
	return t6d
}

func (m Matrix) SliceRow(r int) vec.Vector {
	slice := m[r]
	return slice
}

func (m Matrix) SliceColumn(fix int) vec.Vector {
	slice := vec.Zeros(m.Rows())
	for i := 0; i < m.Rows(); i++ {
		slice[i] = m[i][fix]
	}
	return slice
}

func (m Matrix) String() string {
	str := "[\n"
	for i := 0; i < m.Rows(); i++ {
		str += fmt.Sprintf("  %v,\n", m.SliceRow(i))
	}
	str += "]"
	return str
}

func (m Matrix) Col2Img(shape []int, fh, fw, stride, pad int) Tensor4D {
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

func Zeros(rows int, cols int) Matrix {
	mat := make([]vec.Vector, rows)
	for i := 0; i < rows; i++ {
		mat[i] = vec.Zeros(cols)
	}
	return mat
}

func ZerosLike(x Matrix) Matrix {
	return Zeros(x.Rows(), x.Columns())
}

func NewRandnMatrix(rows, cols int) Matrix {
	if rows == 0 || cols == 0 {
		return nil
	}
	vec := vec.Randn(rows * cols)
	return NewMatrix(vec, rows, cols)
}

func NotEqual(m1, m2 Matrix) bool {
	return !Equal(m1, m2)
}

func Equal(m1, m2 Matrix) bool {
	if m1.Rows() == m2.Rows() && m1.Columns() == m2.Columns() {
		for i := 0; i < m1.Rows(); i++ {
			if vec.NotEqual(m1[i], m2[i]) {
				return false
			}
		}
		return true
	}
	return false
}

func dotPart(i int, a, b, c Matrix, ch chan int) {
	ac := a.Columns()
	bc := b.Columns()
	for j := 0; j < bc; j++ {
		part := 0.0
		for k := 0; k < ac; k++ {
			part += a[i][k] * b[k][j]
		}
		c[i][j] = part
	}
	ch <- i
}

func Dot(m1, m2 Matrix) Matrix {
	if m1.Columns() != m2.Rows() {
		return nil
	}
	m3 := Zeros(m1.Rows(), m2.Columns())
	ch := make(chan int)
	for i := 0; i < m1.Rows(); i++ {
		go dotPart(i, m1, m2, m3, ch)
	}
	for i := 0; i < m1.Rows(); i++ {
		<-ch
	}
	return m3
}

func Dot2(m1, m2 Matrix) Matrix {
	if m1.Columns() != m2.Rows() {
		return nil
	}
	// fmt.Println(m1.Rows * m2.Columns)
	mat := Zeros(m1.Rows(), m2.Columns())
	for i := 0; i < m1.Columns(); i++ {
		for c := 0; c < m2.Columns(); c++ {
			for r := 0; r < m1.Rows(); r++ {
				// mat[r*m2.Columns+c] += m1.Element(r, i) * m2.Element(i, c)
				mat[r][c] += m1[r][i] * m2[i][c]
			}
		}
	}
	return mat
}

func isTheSameShape(m1, m2 Matrix) bool {
	if m1.Columns() == m2.Columns() && m1.Rows() == m2.Rows() {
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

func matMat(a Arithmetic, m1, m2 Matrix) Matrix {
	if !isTheSameShape(m1, m2) {
		// 片方がベクトル(1行多列)だった場合
		if m1.Rows() == 1 || m2.Rows() == 1 {
			if m1.Columns() != m2.Columns() {
				return nil
			}
			vector := Zeros(m1.Rows(), m1.Columns())
			for r := 0; r < m1.Rows(); r++ {
				for c := 0; c < m2.Columns(); c++ {
					switch a {
					case ADD:
						vector[r][c] = m1[r][c] + m2[0][c]
					case SUB:
						vector[r][c] = m1[r][c] - m2[0][c]
					case MUL:
						vector[r][c] = m1[r][c] * m2[0][c]
					case DIV:
						vector[r][c] = m1[r][c] / m2[0][c]
					}
				}
			}
			return vector
		}
	} else {
		vector := Zeros(m1.Rows(), m1.Columns())
		for i, v := range m1 {
			switch a {
			case ADD:
				vector[i] = vec.Add(v, m2[i])
			case SUB:
				vector[i] = vec.Sub(v, m2[i])
			case MUL:
				vector[i] = vec.Mul(v, m2[i])
			case DIV:
				vector[i] = vec.Div(v, m2[i])
			}
		}
		return vector
	}
	return nil
}

func matVec(a Arithmetic, m1 Matrix, m2 vec.Vector) Matrix {
	if m1.Columns() != len(m2) {
		return nil
	}
	vector := Zeros(m1.Rows(), m1.Columns())
	for r := 0; r < m1.Rows(); r++ {
		for c := 0; c < len(m2); c++ {
			switch a {
			case ADD:
				vector[r][c] = m1[r][c] + m2[c]
			case SUB:
				vector[r][c] = m1[r][c] - m2[c]
			case MUL:
				vector[r][c] = m1[r][c] * m2[c]
			case DIV:
				vector[r][c] = m1[r][c] / m2[c]
			}
		}
	}
	return vector
}

func matFloat(a Arithmetic, m1 Matrix, m2 float64) Matrix {
	vector := ZerosLike(m1)
	for i, v := range m1 {
		switch a {
		case ADD:
			vector[i] = vec.Add(v, m2)
		case SUB:
			vector[i] = vec.Sub(v, m2)
		case MUL:
			vector[i] = vec.Mul(v, m2)
		case DIV:
			vector[i] = vec.Div(v, m2)
		}
	}
	return vector
}

func vecMat(a Arithmetic, m1 vec.Vector, m2 Matrix) Matrix {
	if m2.Columns() != len(m1) {
		return nil
	}
	vector := Zeros(m2.Rows(), m2.Columns())
	for r := 0; r < m2.Rows(); r++ {
		for c := 0; c < len(m1); c++ {
			switch a {
			case ADD:
				vector[r][c] = m1[c] + m2[r][c]
			case SUB:
				vector[r][c] = m1[c] - m2[r][c]
			case MUL:
				vector[r][c] = m1[c] * m2[r][c]
			case DIV:
				vector[r][c] = m1[c] / m2[r][c]
			}
		}
	}
	return vector
}

func floatMat(a Arithmetic, m1 float64, m2 Matrix) Matrix {
	vector := ZerosLike(m2)
	for i, v := range m2 {
		switch a {
		case ADD:
			vector[i] = vec.Add(m1, v)
		case SUB:
			vector[i] = vec.Sub(m1, v)
		case MUL:
			vector[i] = vec.Mul(m1, v)
		case DIV:
			vector[i] = vec.Div(m1, v)
		}
	}
	return vector
}

func Add(x1, x2 interface{}) Matrix {
	if x1v, ok := x1.(Matrix); ok {
		switch x2v := x2.(type) {
		case Matrix:
			return matMat(ADD, x1v, x2v)
		case vec.Vector:
			return matVec(ADD, x1v, x2v)
		case float64:
			return matFloat(ADD, x1v, x2v)
		case int:
			return matFloat(ADD, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(Matrix); ok {
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
func Sub(x1, x2 interface{}) Matrix {
	if x1v, ok := x1.(Matrix); ok {
		switch x2v := x2.(type) {
		case Matrix:
			return matMat(SUB, x1v, x2v)
		case vec.Vector:
			return matVec(SUB, x1v, x2v)
		case float64:
			return matFloat(SUB, x1v, x2v)
		case int:
			return matFloat(SUB, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(Matrix); ok {
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

func Mul(x1, x2 interface{}) Matrix {
	if x1v, ok := x1.(Matrix); ok {
		switch x2v := x2.(type) {
		case Matrix:
			return matMat(MUL, x1v, x2v)
		case vec.Vector:
			return matVec(MUL, x1v, x2v)
		case float64:
			return matFloat(MUL, x1v, x2v)
		case int:
			return matFloat(MUL, x1v, float64(x2v))

		}
	} else if x2v, ok := x2.(Matrix); ok {
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

func Div(x1, x2 interface{}) Matrix {
	if x1v, ok := x1.(Matrix); ok {
		switch x2v := x2.(type) {
		case Matrix:
			return matMat(DIV, x1v, x2v)
		case vec.Vector:
			return matVec(DIV, x1v, x2v)
		case float64:
			return matFloat(DIV, x1v, x2v)
		case int:
			return matFloat(DIV, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(Matrix); ok {
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

func Sigmoid(m Matrix) Matrix {
	mat := ZerosLike(m)
	for i, v := range m {
		mat[i] = vec.Sigmoid(v)
	}
	return mat
}

func Relu(m Matrix) Matrix {
	mat := ZerosLike(m)
	for i, v := range m {
		mat[i] = vec.Relu(v)
	}
	return mat
}

func Log(m Matrix) Matrix {
	mat := ZerosLike(m)
	for i, v := range m {
		mat[i] = vec.Log(v)
	}
	return mat
}

func MeanAll(m Matrix) float64 {
	return SumAll(m) / float64(len(m))
}

func SumAll(m Matrix) float64 {
	sum := 0.0
	for _, v := range m {
		sum += vec.Sum(v)
	}
	return sum
}

func Mean(m Matrix, axis int) Matrix {
	if axis == 0 {
		v := vec.Zeros(m.Columns())
		for i := 0; i < m.Columns(); i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Sum(col) / float64(m.Rows())
		}
		return []vec.Vector{v}
	} else if axis == 1 {
		v := vec.Zeros(m.Rows())
		for i := 0; i < m.Rows(); i++ {
			row := m.SliceRow(i)
			v[i] = vec.Sum(row) / float64(m.Columns())
		}
		return []vec.Vector{v}
	}
	return nil
}

func Sum(m Matrix, axis int) Matrix {
	if axis == 0 {
		v := vec.Zeros(m.Columns())
		for i := 0; i < m.Columns(); i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Sum(col)
		}
		return []vec.Vector{v}
	} else if axis == 1 {
		v := vec.Zeros(m.Rows())
		for i := 0; i < m.Rows(); i++ {
			row := m.SliceRow(i)
			v[i] = vec.Sum(row)
		}
		return []vec.Vector{v}
	}
	return nil
}

func MaxAll(x Matrix) float64 {
	max := math.SmallestNonzeroFloat64
	for _, v := range x {
		vecMax := vec.Max(v)
		max = math.Max(max, vecMax)
	}
	return max
}

func Max(m Matrix, axis int) vec.Vector {
	if axis == 0 {
		v := vec.Zeros(m.Columns())
		for i := 0; i < m.Columns(); i++ {
			col := m.SliceColumn(i)
			v[i] = vec.Max(col)
		}
		return v
	} else if axis == 1 {
		v := vec.Zeros(m.Rows())
		for i := 0; i < m.Rows(); i++ {
			row := m.SliceRow(i)
			v[i] = vec.Max(row)
		}
		return v
	}
	return nil
}

func ArgMaxAll(x Matrix) int {
	max := math.SmallestNonzeroFloat64
	maxIndex := 0
	for i, v := range x {
		argMax := float64(vec.ArgMax(v))
		if max != math.Max(max, argMax) {
			maxIndex = i
			max = math.Max(max, argMax)
		}
	}
	return maxIndex
}

func ArgMax(m Matrix, axis int) []int {
	if axis == 0 {
		v := make([]int, m.Columns())
		for i := 0; i < m.Columns(); i++ {
			col := m.SliceColumn(i)
			v[i] = vec.ArgMax(col)
		}
		return v
	} else if axis == 1 {
		v := make([]int, m.Rows())
		for i := 0; i < m.Rows(); i++ {
			row := m.SliceRow(i)
			v[i] = vec.ArgMax(row)
		}
		return v
	}
	return nil
}

func Pow(x Matrix, p float64) Matrix {
	pow := ZerosLike(x)
	for i, v := range x {
		pow[i] = vec.Pow(v, p)
	}
	return pow
}

func Sqrt(x Matrix) Matrix {
	sqrt := ZerosLike(x)
	for i, v := range x {
		sqrt[i] = vec.Sqrt(v)
	}
	return sqrt
}

func Abs(x Matrix) Matrix {
	abs := ZerosLike(x)
	for i, v := range x {
		abs[i] = vec.Abs(v)
	}
	return abs
}

func Exp(x Matrix) Matrix {
	exp := ZerosLike(x)
	for i, v := range x {
		exp[i] = vec.Exp(v)
	}
	return exp
}

func Softmax(x Matrix) Matrix {
	xt := x.T()
	sub := Sub(xt, Max(xt, 0))
	expX := Exp(sub)
	sumExpX := Sum(expX, 0)
	softmax := Div(expX, sumExpX)
	return softmax.T()
}

func CrossEntropyError(y, t Matrix) float64 {
	r := vec.Zeros(y.Rows())
	for i := 0; i < y.Rows(); i++ {
		yRow := y.SliceRow(i)
		tRow := t.SliceRow(i)
		r[i] = vec.CrossEntropyError(yRow, tRow)
	}
	return vec.Sum(r) / float64(y.Rows())
}

func NumericalGradient(f func(vec.Vector) float64, x Matrix) Matrix {
	mat := ZerosLike(x)
	for i, v := range x {
		mat[i] = vec.NumericalGradient(f, v)

	}
	return mat
}
