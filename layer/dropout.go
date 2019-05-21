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
func (d *Dropout) Forward(x *num.Matrix, trainFlg bool) *num.Matrix {
	if trainFlg {
		out := num.ZerosLike(x)
		rand, _ := num.NewRandnMatrix(x.Shape())
		d.Mask = make([]bool, len(rand.Vector))
		for i, v := range rand.Vector {
			if v > d.Ratio {
				d.Mask[i] = true
				out.Vector[i] = x.Vector[i]
			}
		}
		return out
	}
	return num.Mul(x, 1.0-d.Ratio)
}

func (d *Dropout) Backward(dout *num.Matrix) *num.Matrix {
	doutv := dout.Vector
	dv := vec.ZerosLike(doutv)
	for i, e := range doutv {
		if d.Mask[i] {
			dv[i] = e
		} else {
			dv[i] = 0
		}
	}
	dx := &num.Matrix{
		Vector:  dv,
		Rows:    dout.Rows,
		Columns: dout.Columns,
	}
	return dx

}
