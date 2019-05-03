package layer

import "github.com/naronA/zero_deeplearning/scalar"

type Sigmoid struct {
	Out float64
}

func (s *Sigmoid) forward(x float64) float64 {
	out := scalar.Sigmoid(x)
	s.Out = out
	return out
}

func (s *Sigmoid) backward(dout float64) float64 {
	dx := dout * (1.0 - s.Out) * s.Out
	return dx
}
