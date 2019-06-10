package tensor

import (
	"github.com/naronA/zero_deeplearning/vec"
)

func (t *Tensor) SliceMatRow(r int) *Tensor {
	if len(t.Shape) != 2 {
		panic(t)
	}
	m := t.Mat
	slice := m.Vector[r*m.Columns : (r+1)*m.Columns]
	return &Tensor{
		Vec:   slice,
		Shape: []int{len(slice)},
	}
}

func (t *Tensor) SliceMatColumn(c int) *Tensor {
	if len(t.Shape) == 2 {
		panic(t)
	}
	m := t.Mat
	slice := vec.Zeros(m.Rows)
	for i := 0; i < m.Rows; i++ {
		slice[i] = m.Element(i, c)
	}
	return &Tensor{
		Vec:   slice,
		Shape: []int{len(slice)},
	}
}

func (t *Tensor) Slice6DTo4D(x, y int) *Tensor {
	if len(t.Shape) == 6 {
		t6d := t.T6D
		t4d := make(Tensor4D, 0, len(t6d))
		for _, ncolT5d := range t6d {
			t3d := make(Tensor3D, 0, len(ncolT5d))
			for _, ncolT4d := range ncolT5d {
				ncolMat := ncolT4d[x][y]
				t3d = append(t3d, ncolMat)
			}
			t4d = append(t4d, t3d)
		}
		n, c, h, w := t4d.Shape()
		return &Tensor{T4D: t4d, Shape: []int{n, c, h, w}}
	}
	panic(t)
}

func (t *Tensor) StrideSlice(y, yMax, x, xMax, stride int) *Tensor4DSlice {
	indLen := 0
	for _, imgT3D := range t.T4D {
		for k := 0; k < len(imgT3D); k++ {
			for i := y; i < yMax; i += stride {
				for j := x; j < xMax; j += stride {
					indLen++
				}
			}
		}
	}

	indices := make([]*Tensor4DIndex, 0, indLen)
	totalRows := (yMax - y) / stride
	totalColumns := (xMax - x) / stride
	for n, imgT3D := range t.T4D {
		for c := range imgT3D {
			for i := y; i < yMax; i += stride {
				for j := x; j < xMax; j += stride {
					index := &Tensor4DIndex{
						N: n,
						C: c,
						H: i,
						W: j,
					}
					indices = append(indices, index)
				}
			}
		}
	}
	n, c, _, _ := t.T4D.Shape()
	return &Tensor4DSlice{
		Actual:   t.T4D,
		Indices:  indices,
		NewShape: []int{n, c, totalRows, totalColumns},
	}
}

func (t *Tensor) SliceT4D(y, yMax, x, xMax int) *Tensor {
	if len(t.Shape) == 4 {
		t4dslice := t.StrideSlice(y, yMax, x, xMax, 1)
		sliced := t4dslice.ToTensor4D()
		n, c, h, w := sliced.Shape()
		return &Tensor{
			T4D:   t4dslice.ToTensor4D(),
			Shape: []int{n, c, h, w},
		}
	}
	panic(t)
}
