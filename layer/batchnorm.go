package layer

import (
	"github.com/naronA/zero_deeplearning/mat"
)

type BatchNormalization struct {
	Gamma       float64 // パラメタ
	Beta        float64 // パラメタ
	Dgamma      *mat.Matrix
	Dbeta       *mat.Matrix
	Momentum    float64 // パラメタ
	BatchSize   int
	InputShape  func() (int, int)
	RunningMean *mat.Matrix
	RunningVar  *mat.Matrix
	Xc          *mat.Matrix
	Xn          *mat.Matrix
	Std         *mat.Matrix
}

/*
class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
*/
func NewBatchNorimalization(gamma, beta float64) *BatchNormalization {
	return &BatchNormalization{
		Gamma: gamma,
		Beta:  beta,
	}
}

/*

   def forward(self, x, train_flg=True):
       self.input_shape = x.shape
       if x.ndim != 2:
           N, C, H, W = x.shape
           x = x.reshape(N, -1)

       out = self.__forward(x, train_flg)

       return out.reshape(*self.input_shape)
*/
func (b *BatchNormalization) Forward(x *mat.Matrix, trainFlg bool) *mat.Matrix {
	b.InputShape = x.Shape
	// if self.running_mean is None:
	//     N, D = x.shape
	//     self.running_mean = np.zeros(D)
	//     self.running_var = np.zeros(D)

	// 初期化
	if b.RunningMean == nil {
		_, D := x.Shape()
		b.RunningMean = mat.Zeros(1, D)
		b.RunningVar = mat.Zeros(1, D)
	}
	// if train_flg:
	//     mu = x.mean(axis=0)
	//     xc = x - mu
	//     var = np.mean(xc**2, axis=0)
	//     std = np.sqrt(var + 10e-7)
	//     xn = xc / std
	//
	//     self.batch_size = x.shape[0]
	//     self.xc = xc
	//     self.xn = xn
	//     self.std = std
	//     self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
	//     self.running_var = self.momentum * self.running_var + (1-self.momentum) * var

	if trainFlg {
		mu := mat.Mean(x, 0)
		xc := mat.Sub(x, mu)
		vari := mat.Mean(mat.Pow(xc, 2), 0)
		std := mat.Sqrt(mat.Add(vari, 10e-7))
		xn := mat.Div(xc, std)

		// 誤差逆伝搬のために状態を保存
		b.BatchSize, _ = x.Shape()
		b.Xc = xc
		b.Xn = xn
		b.Std = std
		b.RunningMean = mat.Add(mat.Mul(b.RunningMean, b.Momentum), mat.Mul((1.0-b.Momentum), mu))
		b.RunningVar = mat.Add(mat.Mul(b.RunningVar, b.Momentum), mat.Mul((1.0-b.Momentum), vari))
		out := mat.Add(mat.Mul(b.Gamma, xn), b.Beta)
		return out
	}
	//  else:
	//      xc = x - self.running_mean
	//      xn = xc / ((np.sqrt(self.running_var + 10e-7)))
	//  out = self.gamma * xn + self.beta
	//  return out

	xc := mat.Sub(x, b.RunningMean)
	xn := mat.Div(xc, mat.Add(mat.Sqrt(b.RunningVar), 10e-7))
	out := mat.Add(mat.Mul(b.Gamma, xn), b.Beta)
	return out.Reshape(b.InputShape())
}

/*
   def backward(self, dout):
       if dout.ndim != 2:
           N, C, H, W = dout.shape
           dout = dout.reshape(N, -1)

       dx = self.__backward(dout)

       dx = dx.reshape(*self.input_shape)
       return dx

   def __backward(self, dout):
       dbeta = dout.sum(axis=0)
       dgamma = np.sum(self.xn * dout, axis=0)
       dxn = self.gamma * dout
       dxc = dxn / self.std
       dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
       dvar = 0.5 * dstd / self.std
       dxc += (2.0 / self.batch_size) * self.xc * dvar
       dmu = np.sum(dxc, axis=0)
       dx = dxc - dmu / self.batch_size

       self.dgamma = dgamma
       self.dbeta = dbeta

       return dx
*/

func (b *BatchNormalization) Backward(dout *mat.Matrix) *mat.Matrix {
	dBeta := mat.Sum(dout, 0)
	dGamma := mat.Sum(mat.Mul(b.Xn, dout), 0)
	dxn := mat.Mul(b.Gamma, dout)
	dxc := mat.Div(dxn, b.Std)
	dstd := mat.Mul(-1, mat.Sum(mat.Div(mat.Mul(dxn, b.Xc), mat.Mul(b.Std, b.Std)), 0))
	dvar := mat.Div(mat.Mul(0.5, dstd), b.Std)
	dxc = mat.Add(dxc, mat.Mul(mat.Mul(mat.Div(2.0, b.BatchSize), b.Xc), dvar))
	dmu := mat.Sum(dxc, 0)
	dx := mat.Div(mat.Sub(dxc, dmu), b.BatchSize)

	b.Dgamma = dGamma
	b.Dbeta = dBeta

	return dx.Reshape(b.InputShape())
}
