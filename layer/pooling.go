package layer

import (
	"github.com/naronA/zero_deeplearning/num"
)

type Pooling struct {
	PoolH  int
	PoolW  int
	Stride int
	Pad    int

	X      num.Tensor4D
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

func (p *Pooling) Forward(ix interface{}) interface{} {
	x := ix.(num.Tensor4D)
	N, C, H, W := x.Shape()
	outH := 1 + (H-p.PoolH)/p.Stride
	outW := 1 + (W-p.PoolW)/p.Stride
	col := x.Im2Col(p.PoolH, p.PoolW, p.Stride, p.Pad)
	col = col.Reshape(-1, p.PoolH*p.PoolW)

	outVec := num.Max(col, 1)
	outMat := num.NewMatrix(outVec, 1, len(outVec))
	reshaped := outMat.ReshapeTo4D(N, outH, outW, C).Transpose(0, 3, 1, 2)
	p.X = x
	p.ArgMax = num.ArgMax(col, 1)
	return reshaped
}

func (p *Pooling) Backward(idout interface{}) interface{} {
	dout := idout.(num.Tensor4D)
	dout = dout.Transpose(0, 2, 3, 1)
	da, db, dc, dd := dout.Shape()

	poolSize := p.PoolH * p.PoolW
	dmax := num.Zeros(dout.Size(), poolSize)
	for i, v := range dout.Flatten() {
		dmax[i][p.ArgMax[i]] = v
	}
	dmaxT5D := dmax.ReshapeTo5D(da, db, dc, dd, poolSize)
	dm1, dm2, dm3, _, _ := dmaxT5D.Shape()
	dcol := dmaxT5D.ReshapeTo2D(dm1*dm2*dm3, -1)
	a, b, c, d := p.X.Shape()
	dx := dcol.Col2Img([]int{a, b, c, d}, p.PoolH, p.PoolW, p.Stride, p.Pad)
	return dx
}
