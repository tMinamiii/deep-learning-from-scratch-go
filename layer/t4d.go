package layer

import (
	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

type T4DLayer interface {
	Forward(interface{}) interface{}
	Backward(interface{}) interface{}
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

func (c *Convolution) Forward(ix interface{}) interface{} {
	x := ix.(num.Tensor4D)
	FN, _, FH, FW := c.W.Shape()
	N, _, H, W := x.Shape()
	outH := 1 + (H+2*c.Pad-FH)/c.Stride
	outW := 1 + (W+2*c.Pad-FW)/c.Stride

	col := x.Im2Col(FH, FW, c.Stride, c.Pad)
	colW := c.W.ReshapeToMat(FN, -1).T()

	out := num.Add(num.Dot(col, colW), c.B)
	reshape := out.ReshapeTo4D(N, outH, outW, -1)
	trans := reshape.Transpose(0, 3, 1, 2)
	c.X = x
	c.Col = col
	c.ColW = colW
	return trans
}

func (c *Convolution) Backward(idout interface{}) interface{} {
	dout := idout.(num.Tensor4D)
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

/*
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        print(out.shape)

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx
*/
type AffineT4D struct {
	W           *num.Matrix
	B           *num.Matrix
	X           *num.Matrix
	DW          *num.Matrix
	DB          *num.Matrix
	OrigXShapeN int
	OrigXShapeC int
	OrigXShapeH int
	OrigXShapeW int
}

func NewAffineT4D(w, b *num.Matrix) *AffineT4D {
	return &AffineT4D{
		W:           w,
		B:           b,
		OrigXShapeN: 0,
		OrigXShapeC: 0,
		OrigXShapeH: 0,
		OrigXShapeW: 0,
	}
}

func (af *AffineT4D) Forward(x interface{}) interface{} {
	if t4d, ok := x.(num.Tensor4D); ok {
		n, c, h, w := t4d.Shape()
		af.OrigXShapeN = n
		af.OrigXShapeC = c
		af.OrigXShapeH = h
		af.OrigXShapeW = w
		reshapeX := t4d.ReshapeToMat(n, -1)
		af.X = reshapeX
		out := num.Add(num.Dot(reshapeX, af.W), af.B)
		return out
	} else if mat, ok := x.(*num.Matrix); ok {
		af.X = mat
		out := num.Add(num.Dot(mat, af.W), af.B)
		return out
	}
	return nil
}

func (af *AffineT4D) Backward(dout interface{}) interface{} {
	mat := dout.(*num.Matrix)
	if af.OrigXShapeN != 0 {
		dx := num.Dot(mat, af.W.T())
		af.DW = num.Dot(af.X.T(), mat)
		af.DB = num.Sum(mat, 0)
		reshapeX := dx.ReshapeTo4D(af.OrigXShapeN, af.OrigXShapeC, af.OrigXShapeH, af.OrigXShapeW)
		return reshapeX
	}
	dx := num.Dot(mat, af.W.T())
	af.DW = num.Dot(af.X.T(), mat)
	af.DB = num.Sum(mat, 0)
	return dx
}

/*
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
*/
type ReLUT4D struct {
	mask []bool
}

func NewReluT4D() *ReLUT4D {
	return &ReLUT4D{}
}

func (r *ReLUT4D) Forward(x interface{}) interface{} {
	if t4d, ok := x.(num.Tensor4D); ok {
		n, c, h, w := t4d.Shape()
		outt4d := num.ZerosLikeT4D(t4d)
		r.mask = make([]bool, n*c*h*w)
		for i, t3d := range t4d {
			for j, mat := range t3d {
				v := mat.Vector
				for k, e := range v {
					if e <= 0 {
						// r.mask[(i*n+j)*c+k] = true
						r.mask[(i*c+j)*h*w+k] = true
						// r.mask = append(r.mask, true)
						outt4d[i][j].Vector[k] = 0
					} else {
						// r.mask = append(r.mask, false)
						outt4d[i][j].Vector[k] = e
					}
				}
			}
		}
		return outt4d
	} else if mat, ok := x.(*num.Matrix); ok {
		v := mat.Vector
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

		return &num.Matrix{
			Vector:  out,
			Rows:    mat.Rows,
			Columns: mat.Columns,
		}
	}
	return nil
}

func (r *ReLUT4D) Backward(dout interface{}) interface{} {
	if t4d, ok := dout.(num.Tensor4D); ok {
		_, c, h, w := t4d.Shape()
		outt4d := num.ZerosLikeT4D(t4d)
		for i, t3d := range t4d {
			for j, mat := range t3d {
				for k, e := range mat.Vector {
					if r.mask[(i*c+j)*h*w+k] {
						outt4d[i][j].Vector[k] = 0
					} else {
						outt4d[i][j].Vector[k] = e
					}
				}
			}
		}
		return outt4d

	} else if mat, ok := dout.(*num.Matrix); ok {
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
		return dx

	}
	return nil
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

func (so *SoftmaxWithLossT4D) Backward() interface{} {
	batchSize, _, _, _ := so.t.Shape()
	sub := num.SubT4D(so.y, so.t)
	dx := num.DivT4D(sub, batchSize)
	return dx
}
