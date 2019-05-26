package layer

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/num"
)

/*
def __init__(self, pool_h, pool_w, stride=1, pad=0):
    self.pool_h = pool_h
    self.pool_w = pool_w
    self.stride = stride
    self.pad = pad

    self.x = None
    self.arg_max = None
*/
type Pooling struct {
	PoolH  int
	PoolW  int
	Stride int
	Pad    int

	X      num.Tensor4D
	ArgMax []int
}

/*
def forward(self, x):
    N, C, H, W = x.shape
    out_h = int(1 + (H - self.pool_h) / self.stride)
    out_w = int(1 + (W - self.pool_w) / self.stride)

    col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
    col = col.reshape(-1, self.pool_h*self.pool_w)

    arg_max = np.argmax(col, axis=1)
    out = np.max(col, axis=1)
    out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    self.x = x
    self.arg_max = arg_max

    return out
*/
func NewPooling(poolh, poolw, stride, pad int) *Pooling {
	return &Pooling{
		PoolH:  poolh,
		PoolW:  poolw,
		Stride: stride,
		Pad:    pad,
	}
}

func (p *Pooling) Forward(x num.Tensor4D) num.Tensor4D {
	N, C, H, W := x.Shape()
	outH := 1 + (H-p.PoolH)/p.Stride
	outW := 1 + (W-p.PoolW)/p.Stride
	fmt.Println(outH, outW)
	col := x.Im2Col(p.PoolH, p.PoolW, p.Stride, p.Pad)
	fmt.Println(col)
	col = col.Reshape(-1, p.PoolH*p.PoolW)

	outVec := num.Max(col, 1)
	outMat := &num.Matrix{
		Vector:  outVec,
		Rows:    1,
		Columns: len(outVec),
	}
	reshaped := outMat.ReshapeTo4D(N, outH, outW, C).Transpose(0, 3, 1, 2)
	p.X = x
	p.ArgMax = num.ArgMax(col, 1)
	return reshaped
}

/*
def backward(self, dout):
    dout = dout.transpose(0, 2, 3, 1)

    pool_size = self.pool_h * self.pool_w
    dmax = np.zeros((dout.size, pool_size))
    dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (pool_size,))

    dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
    dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

    return dx
*/

func (p *Pooling) Backward(dout num.Tensor4D) num.Tensor4D {
	dout = dout.Transpose(0, 2, 3, 1)
	da, db, dc, dd := dout.Shape()

	poolSize := p.PoolH * p.PoolW
	dmax := num.Zeros(dout.Size(), poolSize)
	for i, v := range dout.Flatten() {
		dmax.Assign(v, i, p.ArgMax[i])
	}
	dmaxT5D := dmax.ReshapeTo5D(da, db, dc, dd, poolSize)
	dm1, dm2, dm3, _, _ := dmaxT5D.Shape()
	dcol := dmaxT5D.ReshapeTo2D(dm1*dm2*dm3, -1)
	a, b, c, d := p.X.Shape()
	dx := dcol.Col2Img([]int{a, b, c, d}, p.PoolH, p.PoolW, p.Stride, p.Pad)
	return dx
}
