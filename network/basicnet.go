package network

import (
	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
)

// BasicNetworkはゼロから作るディープラーニングのp64 Ch3.4.3の
// ニューラルネットワーク
type BasicNetwork struct {
	Network map[string]*mat.Matrix
}

func NewBasicNetwork() *BasicNetwork {
	network := map[string]*mat.Matrix{}

	network["W1"], _ = mat.NewMat64(2, 3, array.Array{
		0.1, 0.3, 0.5,
		0.2, 0.4, 0.6,
	})
	network["b1"], _ = mat.NewMat64(1, 3, array.Array{
		0.1, 0.2, 0.3,
	})
	network["W2"], _ = mat.NewMat64(3, 2, array.Array{
		0.1, 0.4,
		0.2, 0.5,
		0.3, 0.6,
	})
	network["b2"], _ = mat.NewMat64(1, 2, array.Array{
		0.1, 0.2,
	})
	network["W3"], _ = mat.NewMat64(2, 2, array.Array{
		0.1, 0.3,
		0.2, 0.4,
	})
	network["b3"], _ = mat.NewMat64(1, 2, array.Array{
		0.1, 0.2,
	})
	return &BasicNetwork{
		Network: network,
	}
}

func (bn *BasicNetwork) Forward(x *mat.Matrix) array.Array {
	W1 := bn.Network["W1"]
	W2 := bn.Network["W2"]
	W3 := bn.Network["W3"]
	b1 := bn.Network["b1"]
	b2 := bn.Network["b2"]
	b3 := bn.Network["b3"]

	mul1 := x.Dot(W1)
	a1 := mul1.Add(b1)
	z1 := mat.Sigmoid(a1)

	mul2 := z1.Dot(W2)
	a2 := mul2.Add(b2)
	z2 := mat.Sigmoid(a2)

	mul3 := z2.Dot(W3)
	a3 := mul3.Add(b3)
	y := array.IdentityFunction(a3.Array)
	return y
}