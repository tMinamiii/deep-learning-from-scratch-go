package network

import (
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

// BasicNetworkはゼロから作るディープラーニングのp64 Ch3.4.3の
// ニューラルネットワーク
type BasicNetwork struct {
	Network map[string]*mat.Matrix
}

func NewBasicNetwork() *BasicNetwork {
	network := map[string]*mat.Matrix{}

	network["W1"], _ = mat.NewMatrix(2, 3, vec.Vector{
		0.1, 0.3, 0.5,
		0.2, 0.4, 0.6,
	})
	network["b1"], _ = mat.NewMatrix(1, 3, vec.Vector{
		0.1, 0.2, 0.3,
	})
	network["W2"], _ = mat.NewMatrix(3, 2, vec.Vector{
		0.1, 0.4,
		0.2, 0.5,
		0.3, 0.6,
	})
	network["b2"], _ = mat.NewMatrix(1, 2, vec.Vector{
		0.1, 0.2,
	})
	network["W3"], _ = mat.NewMatrix(2, 2, vec.Vector{
		0.1, 0.3,
		0.2, 0.4,
	})
	network["b3"], _ = mat.NewMatrix(1, 2, vec.Vector{
		0.1, 0.2,
	})
	return &BasicNetwork{
		Network: network,
	}
}

func (net *BasicNetwork) Forward(x *mat.Matrix) vec.Vector {
	W1 := net.Network["W1"]
	W2 := net.Network["W2"]
	W3 := net.Network["W3"]
	b1 := net.Network["b1"]
	b2 := net.Network["b2"]
	b3 := net.Network["b3"]

	mul1 := mat.Dot(x, W1)
	a1 := mat.Add(mul1, b1)
	z1 := mat.Sigmoid(a1)

	mul2 := mat.Dot(z1, W2)
	a2 := mat.Add(mul2, b2)
	z2 := mat.Sigmoid(a2)

	mul3 := mat.Dot(z2, W3)
	a3 := mat.Add(mul3, b3)
	y := vec.IdentityFunction(a3.Vector)
	return y
}
