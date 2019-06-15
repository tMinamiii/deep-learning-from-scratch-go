package tensor

import (
	"github.com/naronA/zero_deeplearning/vec"
)

type Layer interface {
	Forward(*Tensor) *Tensor
	Backward(*Tensor) *Tensor
}

type Pooling struct {
	PoolH  int
	PoolW  int
	Stride int
	Pad    int

	X      *Tensor
	ArgMax []int
}

func NewPooling(poolh, poolw, stride, pad int) *Pooling {
	return &Pooling{
		PoolH:  poolh,
		PoolW:  poolw,
		Stride: stride,
		Pad:    pad,
	}
}

func (p *Pooling) Forward(x *Tensor) *Tensor {
	if len(x.Shape) == 4 {
		N := x.Shape[0]
		C := x.Shape[1]
		H := x.Shape[2]
		W := x.Shape[3]
		outH := 1 + (H-p.PoolH)/p.Stride
		outW := 1 + (W-p.PoolW)/p.Stride
		col := x.Im2Col(p.PoolH, p.PoolW, p.Stride, p.Pad)
		col = col.Reshape(-1, p.PoolH*p.PoolW)

		out := col.Max(1)
		reshaped := out.ReshapeTo4D(N, outH, outW, C).Transpose([]int{0, 3, 1, 2})
		p.X = x
		p.ArgMax = col.ArgMax(1)
		return reshaped
	}
	panic(p)
}

func (p *Pooling) Backward(dout *Tensor) *Tensor {
	if len(dout.Shape) == 4 {
		dout = dout.Transpose([]int{0, 2, 3, 1})
		da := dout.Shape[0]
		db := dout.Shape[1]
		dc := dout.Shape[2]
		dd := dout.Shape[3]

		poolSize := p.PoolH * p.PoolW
		dmax := Zeros([]int{dout.Size(), poolSize})
		for i, v := range dout.Flatten() {
			dmax.Assign(v, []int{i, p.ArgMax[i]})
		}
		dmaxT5D := dmax.ReshapeTo5D(da, db, dc, dd, poolSize)
		dm1 := dmaxT5D.Shape[0]
		dm2 := dmaxT5D.Shape[1]
		dm3 := dmaxT5D.Shape[2]
		dcol := dmaxT5D.ReshapeToMat(dm1*dm2*dm3, -1)
		dx := dcol.Col2Img(p.X.Shape, p.PoolH, p.PoolW, p.Stride, p.Pad)
		return dx
	}
	panic(p)
}

type Convolution struct {
	W      *Tensor // 4次元
	B      *Tensor // 2次元
	Stride int
	Pad    int
	// 中間データ（backward時に使用）
	X    *Tensor
	Col  *Tensor
	ColW *Tensor
	// 重み・バイアスパラメータの勾配
	DW *Tensor
	DB *Tensor
}

func NewConvolution(w *Tensor, b *Tensor, stride, pad int) *Convolution {
	return &Convolution{
		W:      w,
		B:      b,
		Stride: stride,
		Pad:    pad,
	}
}

func (c *Convolution) Forward(x *Tensor) *Tensor {
	FN := c.W.Shape[0]
	FH := c.W.Shape[2]
	FW := c.W.Shape[3]
	N := x.Shape[0]
	H := x.Shape[2]
	W := x.Shape[3]
	outH := 1 + (H+2*c.Pad-FH)/c.Stride
	outW := 1 + (W+2*c.Pad-FW)/c.Stride

	col := x.Im2Col(FH, FW, c.Stride, c.Pad)
	colW := c.W.ReshapeToMat(FN, -1).T()
	out := Add(Dot(col, colW), c.B)
	reshape := out.ReshapeTo4D(N, outH, outW, -1)
	trans := reshape.Transpose([]int{0, 3, 1, 2})
	c.X = x
	c.Col = col
	c.ColW = colW
	return trans
}

func (c *Convolution) Backward(dout *Tensor) *Tensor {
	FN := c.W.Shape[0]
	C := c.W.Shape[1]
	FH := c.W.Shape[2]
	FW := c.W.Shape[3]
	doutMat := dout.Transpose([]int{0, 2, 3, 1}).ReshapeToMat(-1, FN)
	c.DB = doutMat.Sum(0)
	dot := Dot(c.Col.T(), doutMat)
	c.DW = dot.T().ReshapeTo4D(FN, C, FH, FW)
	dcol := Dot(doutMat, c.ColW.T())
	i := c.X.Shape[0]
	j := c.X.Shape[1]
	k := c.X.Shape[2]
	l := c.X.Shape[3]
	shape := []int{i, j, k, l}
	dx := dcol.Col2Img(shape, FH, FW, c.Stride, c.Pad)
	return dx
}

type Affine struct {
	W           *Tensor
	B           *Tensor
	X           *Tensor
	DW          *Tensor
	DB          *Tensor
	OrigXShapeN int
	OrigXShapeC int
	OrigXShapeH int
	OrigXShapeW int
}

func NewAffine(w, b *Tensor) *Affine {
	return &Affine{
		W:           w,
		B:           b,
		OrigXShapeN: 0,
		OrigXShapeC: 0,
		OrigXShapeH: 0,
		OrigXShapeW: 0,
	}
}

func (af *Affine) Forward(x *Tensor) *Tensor {
	if len(x.Shape) == 4 {
		n := x.Shape[0]
		c := x.Shape[1]
		h := x.Shape[2]
		w := x.Shape[3]
		af.OrigXShapeN = n
		af.OrigXShapeC = c
		af.OrigXShapeH = h
		af.OrigXShapeW = w
		reshapeX := x.ReshapeToMat(n, -1)
		af.X = reshapeX
		out := Add(Dot(reshapeX, af.W), af.B)
		return out
	} else if len(x.Shape) == 2 {
		af.X = x
		out := Add(Dot(x, af.W), af.B)
		return out
	}
	panic(af)
}

func (af *Affine) Backward(dout *Tensor) *Tensor {
	if af.OrigXShapeN != 0 {
		dx := Dot(dout, af.W.T())
		af.DW = Dot(af.X.T(), dout)
		af.DB = dout.Sum(0)
		reshapeX := dx.ReshapeTo4D(af.OrigXShapeN, af.OrigXShapeC, af.OrigXShapeH, af.OrigXShapeW)
		return reshapeX
	}
	dx := Dot(dout, af.W.T())
	af.DW = Dot(af.X.T(), dout)
	af.DB = dout.Sum(0)
	return dx
}

type ReLU struct {
	mask []bool
}

func NewRelu() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *Tensor) *Tensor {
	if len(x.Shape) == 4 {
		n := x.Shape[0]
		c := x.Shape[1]
		h := x.Shape[2]
		w := x.Shape[3]
		outt4d := ZerosLike(x)
		r.mask = make([]bool, n*c*h*w)
		for i, t3d := range x.T4D {
			for j, mat := range t3d {
				v := mat.Vector
				for k, e := range v {
					if e <= 0 {
						// r.mask[(i*n+j)*c+k] = true
						r.mask[(i*c+j)*h*w+k] = true
						// r.mask = append(r.mask, true)
						outt4d.T4D[i][j].Vector[k] = 0
					} else {
						// r.mask = append(r.mask, false)
						outt4d.T4D[i][j].Vector[k] = e
					}
				}
			}
		}
		return outt4d
	} else if len(x.Shape) == 2 {
		v := x.Mat.Vector
		r.mask = make([]bool, len(v))
		out := vec.ZerosLike(v)
		for i, e := range v {
			if e <= 0 {
				r.mask[i] = true
				out[i] = 0
			} else {
				out[i] = e
			}
		}
		mat := &Matrix{
			Vector:  out,
			Rows:    x.Mat.Rows,
			Columns: x.Mat.Columns,
		}
		return &Tensor{
			Mat:   mat,
			Shape: []int{mat.Rows, mat.Columns},
		}
	}
	return nil
}

func (r *ReLU) Backward(dout *Tensor) *Tensor {
	if len(dout.Shape) == 4 {
		c := dout.Shape[1]
		h := dout.Shape[2]
		w := dout.Shape[3]
		outt4d := ZerosLike(dout)
		for i, t3d := range dout.T4D {
			for j, mat := range t3d {
				for k, e := range mat.Vector {
					if r.mask[(i*c+j)*h*w+k] {
						outt4d.T4D[i][j].Vector[k] = 0
					} else {
						outt4d.T4D[i][j].Vector[k] = e
					}
				}
			}
		}
		return outt4d
	} else if len(dout.Shape) == 2 {
		v := dout.Mat.Vector
		dv := vec.ZerosLike(v)
		for i, e := range v {
			if r.mask[i] {
				dv[i] = 0

			} else {
				dv[i] = e
			}
		}
		mat := &Matrix{
			Vector:  dv,
			Rows:    dout.Mat.Rows,
			Columns: dout.Mat.Columns,
		}
		dx := &Tensor{
			Mat:   mat,
			Shape: []int{mat.Rows, mat.Columns},
		}
		return dx

	}
	return nil
}

type SoftmaxWithLoss struct {
	loss float64
	y    *Tensor
	t    *Tensor
}

func NewSfotmaxWithLoss() *SoftmaxWithLoss {
	return &SoftmaxWithLoss{}
}

func (so *SoftmaxWithLoss) Forward(x, t *Tensor) float64 {
	so.t = t
	so.y = x.Softmax()
	so.loss = so.y.CrossEntropyError(so.t)
	return so.loss
}

func (so *SoftmaxWithLoss) Backward() *Tensor {
	batchSize := so.t.Shape[0]
	sub := Sub(so.y, so.t)
	dx := Div(sub, &Tensor{Val: float64(batchSize)})
	return dx
}
