package network

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

// TestSimpleNetwork は、p111のSimpleNetの精度と損失関数計算の動作確認をす
func TestSimpleNetwork(t *testing.T) {
	w := num.NewMatrix(vec.Vector{
		0.47355232, 0.9977393, 0.84668094,
		0.85557411, 0.03563661, 0.69422093,
	}, 2, 3)
	x := num.NewMatrix(vec.Vector{0.6, 0.9}, 1, 2)

	sn := NewSimpleNet(w)
	actualPredict := sn.Predict(x)
	expectedPredict := num.NewMatrix(vec.Vector{
		1.054148091, 0.630716529, 1.132807401,
	}, 1, 3)
	if num.NotEqual(actualPredict, expectedPredict) {
		fmt.Println(actualPredict, expectedPredict)
		t.Fail()
	}
	ta := num.NewMatrix(vec.Vector{0, 0, 1}, 1, 3)
	actualLoss := sn.Loss(x, ta)
	expectedLoss := 0.9280682857864075
	if actualLoss != expectedLoss {
		fmt.Println(actualLoss, expectedLoss)
		t.Fail()
	}

}

func calcNetwork(wvec vec.Vector) float64 {
	w := num.NewMatrix(wvec, 2, 3)
	x := num.NewMatrix(vec.Vector{0.6, 0.9}, 1, 2)

	sn := NewSimpleNet(w)
	t := num.NewMatrix(vec.Vector{0, 0, 1}, 1, 3)
	return sn.Loss(x, t)
}

// TestSimpleNetwork は、p111のSimpleNetの勾配計算の動作確認をす
func TestSimpleNetworkGradient(t *testing.T) {
	w := num.NewMatrix(vec.Vector{
		0.47355232, 0.9977393, 0.84668094,
		0.85557411, 0.03563661, 0.69422093,
	}, 2, 3)
	actual := num.NumericalGradient(calcNetwork, w)
	expected, _ := num.NewMatrix(vec.Vector{
		0.2192475712392561, 0.14356242984070455, -0.3628100010055757,
		0.3288713569016277, 0.21534364482433954, -0.5442150014750569,
	}, 2, 3)
	if num.NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}
