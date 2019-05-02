package network

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
)

// TestSimpleNetwork は、p111のSimpleNetの精度と損失関数計算の動作確認をす
func TestSimpleNetwork(t *testing.T) {
	w, err := mat.NewMat64(2, 3, array.Array{
		0.47355232, 0.9977393, 0.84668094,
		0.85557411, 0.03563661, 0.69422093,
	})
	if err != nil {
		panic(err)
	}
	x, err := mat.NewMat64(1, 2, array.Array{0.6, 0.9})
	if err != nil {
		panic(err)
	}

	sn := NewSimpleNet(w)
	actualPredict := sn.Predict(x)
	expectedPredict, _ := mat.NewMat64(1, 3, array.Array{
		1.054148091, 0.630716529, 1.132807401,
	})
	if actualPredict.NotEqual(expectedPredict) {
		fmt.Println(actualPredict, expectedPredict)
		t.Fail()
	}
	ta, _ := mat.NewMat64(1, 3, array.Array{0, 0, 1})
	actualLoss := sn.Loss(x, ta)
	expectedLoss := 0.9280682857864075
	if actualLoss != expectedLoss {
		fmt.Println(actualLoss, expectedLoss)
		t.Fail()
	}

}

func calcNetwork(wArray array.Array) float64 {
	w, err := mat.NewMat64(2, 3, wArray)
	if err != nil {
		panic(err)
	}
	x, err := mat.NewMat64(1, 2, array.Array{0.6, 0.9})
	if err != nil {
		panic(err)
	}

	sn := NewSimpleNet(w)
	t, _ := mat.NewMat64(1, 3, array.Array{0, 0, 1})
	return sn.Loss(x, t)
}

// TestSimpleNetwork は、p111のSimpleNetの勾配計算の動作確認をす
func TestSimpleNetworkGradient(t *testing.T) {
	w, err := mat.NewMat64(2, 3, array.Array{
		0.47355232, 0.9977393, 0.84668094,
		0.85557411, 0.03563661, 0.69422093,
	})
	if err != nil {
		panic(err)
	}
	actual := mat.NumericalGradient(calcNetwork, w)
	expected, _ := mat.NewMat64(2, 3, array.Array{
		0.2192475712392561, 0.14356242984070455, -0.3628100010055757,
		0.3288713569016277, 0.21534364482433954, -0.5442150014750569,
	})
	if actual.NotEqual(expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}
