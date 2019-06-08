package layer

import (
	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

type Dropout struct {
	Mask  []bool
	Ratio float64
}

func NewDropout(ratio float64) *Dropout {
	return &Dropout{
		Mask:  nil,
		Ratio: ratio,
	}
}

// def __init__(self, dropout_ratio=0.5):
//     self.dropout_ratio = dropout_ratio
//     self.mask = None
//
// def forward(self, x, train_flg=True):
//     if train_flg:
//         self.mask = np.random.rand(*x.shape) > self.dropout_ratio
//         return x * self.mask
//     else:
//         return x * (1.0 - self.dropout_ratio)
//
// def backward(self, dout):
//     return dout * self.mask
func (d *Dropout) Forward(x num.Matrix, trainFlg bool) num.Matrix {
	if trainFlg {
		out := num.ZerosLike(x)
		rand := num.NewRandnMatrix(x.Shape())
		d.Mask = make([]bool, len(rand.Flatten()))
		for i, v := range rand.Flatten() {
			if v > d.Ratio {
				d.Mask[i] = true
				r := i / x.Columns()
				c := i % x.Columns()
				out[r][c] = x[r][c]
			}
		}
		return out
	}
	return num.Mul(x, 1.0-d.Ratio)
}

func (d *Dropout) Backward(dout num.Matrix) num.Matrix {
	doutv := dout.Flatten()
	dv := vec.ZerosLike(doutv)
	for i, e := range doutv {
		if d.Mask[i] {
			dv[i] = e
		} else {
			dv[i] = 0
		}
	}
	dx := num.Zeros(dout.Rows(), dout.Columns())
	for i := 0; i < dout.Rows(); i++ {
		for j := 0; j < dout.Columns(); j++ {
			dx[i][j] = dv[i*dout.Columns()+j]
		}
	}
	return dx
}
