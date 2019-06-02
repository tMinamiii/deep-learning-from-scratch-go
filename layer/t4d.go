package layer

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

type T4DLayer interface {
	Forward(num.Tensor4D) num.Tensor4D
	Backward(num.Tensor4D) num.Tensor4D
}

type Convolution struct {
	W      num.Tensor4D // 4次元
	B      *num.Matrix  // 3次元
	Stride int
	Pad    int
	// 中間データ（backward時に使用）
	X    num.Tensor4D
	Col  *num.Matrix
	ColW *num.Matrix
	// 重み・バイアスパラメータの勾配
	DW num.Tensor4D
	DB *num.Matrix
}

func NewConvolution(w num.Tensor4D, b *num.Matrix, stride, pad int) *Convolution {
	return &Convolution{
		W:      w,
		B:      b,
		Stride: stride,
		Pad:    pad,
	}
}

func (c *Convolution) Forward(x num.Tensor4D) num.Tensor4D {
	FN, _, FH, FW := c.W.Shape()
	N, _, H, W := x.Shape()
	outH := 1 + (H+2*c.Pad-FH)/c.Stride
	outW := 1 + (W+2*c.Pad-FW)/c.Stride

	col := x.Im2Col(FH, FW, c.Stride, c.Pad)
	colW := c.W.ReshapeToMat(FN, -1).T()

	fmt.Println(col.Shape())
	fmt.Println(colW.Shape())
	out := num.Add(num.Dot(col, colW), c.B)
	reshape := out.ReshapeTo4D(N, outH, outW, -1)
	trans := reshape.Transpose(0, 3, 1, 2)
	c.X = x
	c.Col = col
	c.ColW = colW
	return trans
}

func (c *Convolution) Backward(dout num.Tensor4D) num.Tensor4D {
	FN, C, FH, FW := c.W.Shape()
	doutMat := dout.Transpose(0, 2, 3, 1).ReshapeToMat(-1, FN)
	c.DB = num.Sum(doutMat, 0)
	dot := num.Dot(c.Col.T(), doutMat)
	c.DW = dot.T().ReshapeTo4D(FN, C, FH, FW)
	dcol := num.Dot(doutMat, c.ColW.T())
	i, j, k, l := c.X.Shape()
	shape := []int{i, j, k, l}
	dx := dcol.Col2Img(shape, FH, FW, c.Stride, c.Pad)
	return dx
}

type AffineT4D struct {
	W  *num.Matrix
	B  *num.Matrix
	X  num.Tensor4D
	DW num.Tensor4D
	DB *num.Matrix
}

func NewAffineT4D(w, b *num.Matrix) *AffineT4D {
	return &AffineT4D{
		W: w,
		B: b,
	}
}

func (af *AffineT4D) Forward(x num.Tensor4D) *num.Matrix {
	af.X = x
	N, _, _, _ := x.Shape()
	matX := x.ReshapeToMat(N, -1)
	out := num.Add(num.Dot(matX, af.W), af.B)
	return out
}

func (af *AffineT4D) Backward(dout num.Tensor4D) num.Tensor4D {
	dx := num.ZerosLikeT4D(dout)
	dw := num.ZerosLikeT4D(af.DW)
	for i, t3d := range dout {
		for j, mat := range t3d {
			dx[i][j] = num.Dot(mat, af.W.T())
			dw[i][j] = num.Dot(af.X[i][j].T(), dout[i][j])
		}
	}
	FN, _, _, _ := dout.Shape()
	doutMat := dout.ReshapeToMat(-1, FN)
	// doutMat := dout.Transpose(0, 2, 3, 1).ReshapeToMat(-1, FN)
	af.DB = num.Sum(doutMat, 0)
	af.DW = dw
	return dx
}

type ReLUT4D struct {
	mask []bool
}

func NewReluT4D() *ReLUT4D {
	return &ReLUT4D{}
}

func (r *ReLUT4D) Forward(x num.Tensor4D) num.Tensor4D {
	outt4d := num.ZerosLikeT4D(x)
	for i, t3d := range x {
		for j, mat := range t3d {
			v := mat.Vector
			r.mask = make([]bool, len(v))
			zeroVec := vec.ZerosLike(v)
			for k, e := range zeroVec {
				if e <= 0 {
					r.mask[k] = true
					zeroVec[k] = 0
				} else {
					zeroVec[k] = e
				}
			}
			out := &num.Matrix{
				Vector:  zeroVec,
				Rows:    mat.Rows,
				Columns: mat.Columns,
			}
			outt4d[i][j] = out
		}
	}
	return outt4d
}

func (r *ReLUT4D) Backward(dout num.Tensor4D) num.Tensor4D {
	outt4d := num.ZerosLikeT4D(dout)
	for i, t3d := range dout {
		for j, mat := range t3d {
			v := mat.Vector
			dv := vec.ZerosLike(v)
			for i, e := range v {
				if r.mask[i] {
					dv[i] = 0
				} else {
					dv[i] = e
				}
			}
			dx := &num.Matrix{
				Vector:  dv,
				Rows:    mat.Rows,
				Columns: mat.Columns,
			}
			outt4d[i][j] = dx
		}
	}
	return outt4d
}

type SoftmaxWithLossT4D struct {
	loss float64
	y    num.Tensor4D
	t    num.Tensor4D
}

func NewSfotmaxWithLossT4D() *SoftmaxWithLossT4D {
	return &SoftmaxWithLossT4D{}
}

func (so *SoftmaxWithLossT4D) Forward(x, t num.Tensor4D) float64 {
	so.t = t
	so.y = num.SoftmaxT4D(x)
	so.loss = num.CrossEntropyErrorT4D(so.y, so.t)
	return so.loss
}

func (so *SoftmaxWithLossT4D) Backward() num.Tensor4D {
	batchSize, _, _, _ := so.t.Shape()
	sub := num.SubT4D(so.y, so.t)
	dx := num.DivT4D(sub, batchSize)
	return dx
}
