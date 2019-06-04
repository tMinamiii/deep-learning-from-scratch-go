package num

import (
	"errors"

	"github.com/naronA/zero_deeplearning/vec"
)

type Tensor4D []Tensor3D

func (t Tensor4D) Size() int {
	n, c, h, w := t.Shape()
	return n * c * h * w
}

func (t Tensor4D) Element(n, c, h, w int) float64 {
	return t[n].Element(c, h, w)
}

func (t Tensor4D) Assign(value float64, n, c, h, w int) {
	t[n].Assign(value, c, h, w)
}

func (t Tensor4D) Flatten() vec.Vector {
	v := vec.Vector{}
	for _, e := range t {
		v = append(v, e.Flatten()...)
	}
	return v
}

func (t Tensor4D) Transpose(a, b, c, d int) Tensor4D {
	w, x, y, z := t.Shape()
	shape := []int{w, x, y, z}
	t4d := ZerosT4D(shape[a], shape[b], shape[c], shape[d])
	for i, e1t3d := range t {
		for j, e2mat := range e1t3d {
			for k := 0; k < e2mat.Rows; k++ {
				for l := 0; l < e2mat.Columns; l++ {
					oldIdx := []int{i, j, k, l}
					idx := make([]int, 4)
					idx[0] = oldIdx[a]
					idx[1] = oldIdx[b]
					idx[2] = oldIdx[c]
					idx[3] = oldIdx[d]
					v := t.Element(i, j, k, l)
					t4d.Assign(v, idx[0], idx[1], idx[2], idx[3])
				}
			}
		}
	}
	return t4d
}

func (t Tensor4D) ReshapeToMat(row, col int) *Matrix {
	size := t.Size()
	if col == -1 {
		col = size / row
	} else if row == -1 {
		row = size / col
	}
	flat := t.Flatten()
	return &Matrix{
		Vector:  flat,
		Rows:    row,
		Columns: col,
	}
}

func (t Tensor4D) Window(x, y, h, w int) Tensor4D {
	newT4D := make(Tensor4D, len(t))
	for i, mat := range t {
		newT4D[i] = mat.Window(x, y, h, w)
	}
	return newT4D
}

func (t Tensor4D) Pad(size int) Tensor4D {
	newT4D := make(Tensor4D, len(t))
	for i, t3d := range t {
		padded := t3d.Pad(size)
		newT4D[i] = padded
	}
	return newT4D
}

func (t Tensor4D) Shape() (int, int, int, int) {
	N := len(t)
	C := t[0].Channels()
	H, W := t[0][0].Shape()
	return N, C, H, W
}

type Tensor4DIndex struct {
	N int
	C int
	H int
	W int
}

type Tensor4DSlice struct {
	Actual   Tensor4D
	Indices  []*Tensor4DIndex
	NewShape []int
}

func (t4s *Tensor4DSlice) ToTensor4D() Tensor4D {
	newT4D := ZerosT4D(t4s.NewShape[0], t4s.NewShape[1], t4s.NewShape[2], t4s.NewShape[3])
	for i, idx := range t4s.Indices {
		val := t4s.Actual[idx.N][idx.C].Element(idx.H, idx.W)
		matrixLength := t4s.NewShape[2] * t4s.NewShape[3]
		newMatIdx := i - idx.C*matrixLength - idx.N*(matrixLength*t4s.NewShape[1])
		newT4D[idx.N][idx.C].Vector[newMatIdx] = val
	}
	return newT4D
}

func AddAssign(t1 *Tensor4DSlice, t2 Tensor4D) {
	t2flat := t2.Flatten()
	for i, idx := range t1.Indices {
		add := t1.Actual[idx.N][idx.C].Element(idx.H, idx.W) + t2flat[i]
		t1.Actual[idx.N][idx.C].Assign(add, idx.H, idx.W)
	}
}
func (t Tensor4D) StrideSlice(y, yMax, x, xMax, stride int) *Tensor4DSlice {
	n, c, _, _ := t.Shape()
	indices := []*Tensor4DIndex{}
	// for i := y; i < yMax; i += stride {
	// 	totalRows++
	// }
	totalRows := (yMax - y) / stride
	totalColumns := (xMax - x) / stride
	for n, imgT3D := range t {
		for c := range imgT3D {
			for i := y; i < yMax; i += stride {
				for j := x; j < xMax; j += stride {
					index := &Tensor4DIndex{n, c, i, j}
					indices = append(indices, index)
				}
			}
		}
	}
	return &Tensor4DSlice{
		Actual:   t,
		Indices:  indices,
		NewShape: []int{n, c, totalRows, totalColumns},
	}
}

func (t Tensor4D) Slice(y, yMax, x, xMax int) Tensor4D {
	return t.StrideSlice(y, yMax, x, xMax, 1).ToTensor4D()
}
func ZerosT4D(n, c, h, w int) Tensor4D {
	t4d := make(Tensor4D, n)
	for i := range t4d {
		t4d[i] = ZerosT3D(c, h, w)
	}
	return t4d
}

func ZerosLikeT4D(x Tensor4D) Tensor4D {
	t4d := make(Tensor4D, len(x))
	for i, v := range x {
		t4d[i] = ZerosLikeT3D(v)
	}
	return t4d
}

func EqualT4D(t1, t2 Tensor4D) bool {
	for i := range t1 {
		if !EqualT3D(t1[i], t2[i]) {
			return false
		}
	}
	return true
}

func (t Tensor4D) Im2Col(fw, fh, stride, pad int) *Matrix {
	// colVecLen := 0
	// for i := 0; i < len(t); i++ {
	// 	colVecLen++
	// }
	// nVLen := 0
	// for _, t3d := range t {
	// 	for x := 0; x <= t3d[0].Columns-fw+2*pad; x += stride {
	// 		for y := 0; y <= t3d[0].Rows-fh+2*pad; y += stride {
	// 			for i := 0; i < len(t3d); i++ {
	// 				nVLen++
	// 			}
	// 		}
	// 	}
	// 	break
	// }
	colVec := vec.Vector{}
	for _, t3d := range t {
		// nV := make(vec.Vector, nVLen)
		nV := vec.Vector{}
		for x := 0; x <= t3d[0].Columns-fw+2*pad; x += stride {
			for y := 0; y <= t3d[0].Rows-fh+2*pad; y += stride {
				for _, ma := range t3d {
					padE := ma.Pad(pad)
					nV = append(nV, padE.Window(x, y, fw, fh).Vector...)
				}
			}
		}
		colVec = append(colVec, nV...)
	}

	// N, C, H, _ := t.Shape()
	// fmt.Println(len(colVec), N*C*H, fw*fh*C)
	col := fw * fh
	row := len(colVec) / col
	// col := len(colVec) / N / C / H
	// fmt.Println(col)
	return &Matrix{
		Vector:  colVec,
		Rows:    row,
		Columns: col,
		// Columns: fw * fh * C,
	}
}

func NewRandnT4D(n, c, h, w int) (Tensor4D, error) {
	if n == 0 || c == 0 || h == 0 || w == 0 {
		return nil, errors.New("row/columns is zero")
	}
	t4d := make(Tensor4D, n)
	for i := 0; i < n; i++ {
		t3d, _ := NewRandnT3D(c, h, w)
		t4d[i] = t3d
	}
	return t4d, nil
}

func calcT4d(a ArithmeticT4D, x1 interface{}, x2 interface{}) Tensor3D {
	switch a {
	case ADDT4D:
		return AddT3D(x1, x2)
	case SUBT4D:
		return SubT3D(x1, x2)
	case MULT4D:
		return MulT3D(x1, x2)
	case DIVT4D:
		return DivT3D(x1, x2)
	}
	return nil
}

func t4dT4d(a ArithmeticT4D, x1 Tensor4D, x2 Tensor4D) Tensor4D {
	mats := make(Tensor4D, len(x1))
	x1mat := x1
	x2mat := x2
	for i := range x1 {
		mats[i] = calcT4d(a, x1mat[i], x2mat[i])
	}
	return mats
}

func t4dAny(a ArithmeticT4D, x1 Tensor4D, x2 interface{}) Tensor4D {
	mats := make(Tensor4D, len(x1))
	for i, x1mat := range x1 {
		mats[i] = calcT4d(a, x1mat, x2)
	}
	return mats
}

func anyT4d(a ArithmeticT4D, x1 interface{}, x2 Tensor4D) Tensor4D {
	tensor := ZerosLikeT4D(x2)
	mats := tensor
	for i, x2mat := range x2 {
		mats[i] = calcT4d(a, x1, x2mat)
	}
	return tensor
}

type ArithmeticT4D int

const (
	ADDT4D ArithmeticT4D = iota
	SUBT4D
	MULT4D
	DIVT4D
)

func calcArithmeticT4D(a ArithmeticT4D, x1 interface{}, x2 interface{}) Tensor4D {
	if x1v, ok := x1.(Tensor4D); ok {
		switch x2v := x2.(type) {
		case Tensor4D:
			return t4dT4d(a, x1v, x2v)
		case *Matrix:
		case vec.Vector:
		case float64:
			return t4dAny(a, x1v, x2v)
		case int:
			return t4dAny(a, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(Tensor4D); ok {
		switch x1v := x1.(type) {
		case *Matrix:
		case vec.Vector:
		case float64:
			return anyT4d(a, x1v, x2v)
		case int:
			return anyT4d(a, float64(x1v), x2v)
		}
	}
	return nil
}

func AddT4D(x1 interface{}, x2 interface{}) Tensor4D {
	return calcArithmeticT4D(ADDT4D, x1, x2)
}

func SubT4D(x1 interface{}, x2 interface{}) Tensor4D {
	return calcArithmeticT4D(SUBT4D, x1, x2)
}

func MulT4D(x1 interface{}, x2 interface{}) Tensor4D {
	return calcArithmeticT4D(MULT4D, x1, x2)
}

func DivT4D(x1 interface{}, x2 interface{}) Tensor4D {
	return calcArithmeticT4D(DIVT4D, x1, x2)
}

func SoftmaxT4D(x Tensor4D) Tensor4D {
	t4d := ZerosLikeT4D(x)
	for i, v := range x {
		t4d[i] = SoftmaxT3D(v)
	}
	return t4d
}
func CrossEntropyErrorT4D(y, t Tensor4D) float64 {
	r := vec.Zeros(len(y))
	for i := range y {
		r[i] = CrossEntropyErrorT3D(y[i], t[i])
	}
	return vec.Sum(r) / float64(len(y))
}

func NumericalGradientT4D(f func(vec.Vector) float64, x Tensor4D) Tensor4D {
	result := ZerosLikeT4D(x)
	for i, v := range x {
		result[i] = NumericalGradientT3D(f, v)
	}
	return result
}
func PowT4D(x Tensor4D, p float64) Tensor4D {
	result := ZerosLikeT4D(x)
	for i, v := range x {
		result[i] = PowT3D(v, p)
	}
	return result
}

func SqrtT4D(x Tensor4D) Tensor4D {
	result := ZerosLikeT4D(x)
	for i, v := range x {
		result[i] = SqrtT3D(v)
	}
	return result
}

func DotT4D(x Tensor4D, y *Matrix) Tensor4D {
	result := ZerosLikeT4D(x)
	for i, v := range x {
		result[i] = DotT3D(v, y)
	}
	return result
}
