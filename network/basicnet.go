package network

import (
	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

// BasicNetworkはゼロから作るディープラーニングのp64 Ch3.4.3の
// ニューラルネットワーク
type BasicNetwork struct {
	Network map[string]num.Matrix
}

func NewBasicNetwork() *BasicNetwork {
	network := map[string]num.Matrix{}

	network["W1"] = num.NewMatrix(vec.Vector{
		0.1, 0.3, 0.5,
		0.2, 0.4, 0.6,
	}, 2, 3)
	network["b1"] = num.NewMatrix(vec.Vector{
		0.1, 0.2, 0.3,
	}, 1, 2)
	network["W2"] = num.NewMatrix(vec.Vector{
		0.1, 0.4,
		0.2, 0.5,
		0.3, 0.6,
	}, 3, 2)
	network["b2"] = num.NewMatrix(vec.Vector{
		0.1, 0.2,
	}, 1, 2)
	network["W3"] = num.NewMatrix(vec.Vector{
		0.1, 0.3,
		0.2, 0.4,
	}, 2, 2)
	network["b3"] = num.NewMatrix(vec.Vector{
		0.1, 0.2,
	}, 1, 2)
	return &BasicNetwork{
		Network: network,
	}
}

func (net *BasicNetwork) Forward(x num.Matrix) vec.Vector {
	W1 := net.Network["W1"]
	W2 := net.Network["W2"]
	W3 := net.Network["W3"]
	b1 := net.Network["b1"]
	b2 := net.Network["b2"]
	b3 := net.Network["b3"]

	mul1 := num.Dot(x, W1)
	a1 := num.Add(mul1, b1)
	z1 := num.Sigmoid(a1)

	mul2 := num.Dot(z1, W2)
	a2 := num.Add(mul2, b2)
	z2 := num.Sigmoid(a2)

	mul3 := num.Dot(z2, W3)
	a3 := num.Add(mul3, b3)
	y := vec.IdentityFunction(a3.Flatten())
	return y
}
