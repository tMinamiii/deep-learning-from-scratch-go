package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func (t *Tensor) Col2Img(shape []int, fh, fw, stride, pad int) *Tensor {
	if len(t.Shape) != 2 {
		panic(t)
	}

	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	outH := (H+2*pad-fh)/stride + 1
	outW := (W+2*pad-fw)/stride + 1
	ncol := t.Reshape2DTo6D(N, outH, outW, C, fh, fw).Transpose([]int{0, 3, 4, 5, 1, 2})
	// ncol := m.ReshapeTo6D(N, outH, outW, C, fh, fw).Transpose(0, 3, 4, 5, 1, 2)
	img := Zeros([]int{N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1})
	// img := ZerosT4D(N, C, H+2*pad+stride-1, W+2*pad+stride-1)
	for y := 0; y < fh; y++ {
		yMax := y + stride*outH
		for x := 0; x < fw; x++ {
			xMax := x + stride*outW
			slice := img.StrideSlice(y, yMax, x, xMax, stride)
			ncolSlice := ncol.Slice6DTo4D(y, x)
			AddAssign(slice, ncolSlice)
		}
	}
	return img.SliceT4D(pad, H+pad, pad, W+pad)
}

func (t *Tensor) Im2Col(fw, fh, stride, pad int) *Tensor {
	if len(t.Shape) != 4 {
		panic(t)
	}

	t4d := t.T4D
	nVLen := 0
	for _, t3d := range t4d {
		for x := 0; x <= t3d[0].Columns-fw+2*pad; x += stride {
			for y := 0; y <= t3d[0].Rows-fh+2*pad; y += stride {
				for i := 0; i < len(t3d); i++ {
					nVLen++
				}
			}
		}
	}

	colVec := make(vec.Vector, 0, len(t4d))
	for _, t3d := range t4d {
		nV := make(vec.Vector, 0, nVLen)
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

	col := fw * fh
	row := len(colVec) / col
	return &Tensor{
		Mat: &types.Matrix{
			Vector:  colVec,
			Rows:    row,
			Columns: col,
		},
		Shape: []int{row, col},
	}
}
