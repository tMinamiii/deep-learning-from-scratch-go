package network

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

// TestBasicNetwork はp64-p65のニューラルネットワークの動作確認をします
func TestBasicNetwork(t *testing.T) {
	network := NewBasicNetwork()
	x, _ := mat.NewMatrix(1, 2, vec.Vector{
		1.0, 0.5,
	})
	actual := network.Forward(x)
	expected := vec.Vector{0.3168270764110298, 0.6962790898619668}
	if vec.NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}
