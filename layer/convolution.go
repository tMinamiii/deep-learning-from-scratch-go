package layer

import (
	"github.com/naronA/zero_deeplearning/num"
)

/*
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx



*/
type Convolution struct {
	W      num.Tensor4D // 4次元
	B      num.Tensor4D // 3次元
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

func NewConvolution(w, b num.Tensor4D, stride, pad int) *Convolution {
	return &Convolution{
		W:      w,
		B:      b,
		Stride: stride,
		Pad:    pad,
	}
}

/*
   def forward(self, x):
       FN, C, FH, FW = self.W.shape
       N, C, H, W = x.shape
       out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
       out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

       col = im2col(x, FH, FW, self.stride, self.pad)
       col_W = self.W.reshape(FN, -1).T

       out = np.dot(col, col_W) + self.b
       out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

       self.x = x
       self.col = col
       self.col_W = col_W

       return out
*/
func (c *Convolution) Forward(x num.Tensor4D) interface{} {
	FN, _, FH, FW := c.W.Shape()
	N, _, H, W := x.Shape()
	outH := 1 + (H+2*c.Pad-FH)/c.Stride
	outW := 1 + (W+2*c.Pad-FW)/c.Stride

	col := x.Im2Col(FH, FW, c.Stride, c.Pad)
	colW := c.W.ReshapeToMat(FN, -1).T()
	out := num.Add(num.Dot(col, colW), c.B)
	trans := out.ReshapeTo4D(N, outH, outW, -1).Transpose(0, 3, 1, 2)
	c.X = x
	c.Col = col
	c.ColW = colW
	return trans
}

/*
   def backward(self, dout):
       FN, C, FH, FW = self.W.shape
       dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

       self.db = np.sum(dout, axis=0)
       self.dW = np.dot(self.col.T, dout)
       self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

       dcol = np.dot(dout, self.col_W.T)
       dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

       return dx
*/

func (c *Convolution) Backward(dout num.Tensor4D) interface{} {
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
