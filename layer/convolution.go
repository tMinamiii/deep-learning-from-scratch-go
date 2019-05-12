package layer

import (
	"github.com/naronA/zero_deeplearning/mat"
)

type Convolution struct {
	W      mat.Tensor4D // 4次元
	B      mat.Tensor4D // 3次元
	Stride int
	Pad    int
}

func NewConvolution(w, b mat.Tensor4D, stride, pad int) *Convolution {
	return &Convolution{
		W:      w,
		B:      b,
		Stride: stride,
		Pad:    pad,
	}
}

func (c *Convolution) Forward(x mat.Tensor4D) interface{} {
	FN, C, FH, FW := c.W.Shape()
	N, C, H, W := x.Shape()
	outH := int(1 + (H+2*c.Pad-FH)/c.Stride)
	outW := int(1 + (W+2*c.Pad-FW)/c.Stride)

	col := x.Im2Col(FH, FW, c.Stride, c.Pad)
	colW := c.W.ReshapeToMat(FN, -1).T()
	out := mat.Add(mat.Dot(col, colW), c.B)
	out = out.ReshapeTo4D(N, outH, outW, -1).Transpose(0, 3, 1, 2)
	// return out
	return nil
}
