package layer

import "github.com/naronA/zero_deeplearning/mat"

type BatchNormalization struct {
	Gamma       float64
	Beta        float64
	Momentum    float64
	InputShape  func() (int, int)
	RunningMean float64
	RunningVar  float64
	BatchSize   int
	Xc          float64
	Std         float64
	Dgamma      float64
	Dbeta       float64
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
func (b *BatchNormalization) Update(x *mat.Matrix) *mat.Matrix {
	trainFlg := true
	b.InputShape = x.Shape
}

/*
   def __forward(self, x, train_flg):
       if self.running_mean is None:
           N, D = x.shape
           self.running_mean = np.zeros(D)
           self.running_var = np.zeros(D)

       if train_flg:
           mu = x.mean(axis=0)
           xc = x - mu
           var = np.mean(xc**2, axis=0)
           std = np.sqrt(var + 10e-7)
           xn = xc / std

           self.batch_size = x.shape[0]
           self.xc = xc
           self.xn = xn
           self.std = std
           self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
           self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
       else:
           xc = x - self.running_mean
           xn = xc / ((np.sqrt(self.running_var + 10e-7)))

       out = self.gamma * xn + self.beta
       return out
*/
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
