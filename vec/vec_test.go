package vec

import (
	"log"
	"math"
	"testing"
)

func TestVectorMulti(t *testing.T) {
	x := Vector{2, 4, 5}
	w := Vector{1, 2, 3}
	result := x.Mul(w)
	if result == nil {
		t.Fail()
	}
	if !result.Equal(Vector{2, 8, 15}) {
		t.Fail()
	}
	if result.Equal(Vector{1, 4, 10}) {
		t.Fail()
	}
	if result.Equal(Vector{2, 9, 15}) {
		t.Fail()
	}
	if result.Equal(Vector{2, 9}) {
		t.Fail()
	}
}

func TestVectorSum(t *testing.T) {
	ary := Vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	result := Sum(ary)
	if result != 55 {
		t.Fail()
	}
}

func TestVectorMax(t *testing.T) {
	ary := Vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	result := Max(ary)
	if result != 10 {
		t.Fail()
	}
}

func TestVectorDivide(t *testing.T) {
	ary := Vector{1, 2, 3, 4, 5}
	result := ary.Div(10)
	expected := Vector{1.0 / 10.0, 2.0 / 10.0, 3.0 / 10.0, 4.0 / 10.0, 5.0 / 10.0}
	if !result.Equal(expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestVectorSub(t *testing.T) {
	ary := Vector{1, 2, 3, 4, 5}
	result := ary.Sub(2)
	expected := Vector{1 - 2, 2 - 2, 3 - 2, 4 - 2, 5 - 2}
	if !result.Equal(expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestVectorExp(t *testing.T) {
	ary := Vector{1, 2, 3, 4, 5}
	result := Exp(ary)
	expected := Vector{math.Exp(1.0), math.Exp(2.0), math.Exp(3.0), math.Exp(4.0), math.Exp(5.0)}
	if !result.Equal(expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestSoftmax(t *testing.T) {
	actual := Softmax(Vector{-1, 0, 1, 10})
	expected := Vector{1.669860300844509e-05, 4.539150911850783e-05, 0.0001233869144031729, 0.9998145229734698}
	if !actual.Equal(expected) {
		log.Println(actual, expected)
		t.Fail()
	}
	if Sum(actual)-1 != 0 {
		if Sum(actual)-1 > 0 && Sum(actual)-1 > 0.0000001 {
			t.Fail()
		} else if Sum(actual)-1 < 0 && Sum(actual)-1 < -0.0000001 {
			t.Fail()
		}
	}
}

func TestMeanSquaredError1(t *testing.T) {
	k := Vector{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	y := Vector{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0}
	actual := MeanSquaredError(y, k)
	expected := 0.09750000000000003
	if actual != expected {
		log.Println(actual, expected)
		t.Fail()
	}
}

func TestMeanSquaredError2(t *testing.T) {
	k := Vector{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	y := Vector{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0}
	actual := MeanSquaredError(y, k)
	expected := 0.5974999999999999
	if actual != expected {
		log.Println(actual, expected)
		t.Fail()
	}
}

func TestCrossEntropyError1(t *testing.T) {
	k := Vector{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	y := Vector{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0}
	actual := CrossEntropyError(y, k)
	expected := 0.51082545709933802
	if actual != expected {
		log.Println(actual, expected)
		t.Fail()
	}
}

func TestCrossEntropyError2(t *testing.T) {
	k := Vector{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	y := Vector{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0}
	actual := CrossEntropyError(y, k)
	expected := 2.3025840929945458
	if actual != expected {
		log.Println(actual, expected)
		t.Fail()
	}
}

func TestNumericalDiff1(t *testing.T) {
	actual := NumericalGradient(function1, Vector{5})
	expected := Vector{0.1999999999990898}
	if actual.NotEqual(expected) {
		log.Println(actual, expected)
		t.Fail()
	}
}

func TestNumericalDiff2(t *testing.T) {
	actual := NumericalGradient(function1, Vector{10})
	expected := Vector{0.2999999999997449}
	if actual.NotEqual(expected) {
		log.Println(actual, expected)
		t.Fail()
	}
}

func TestGradientDescent1(t *testing.T) {
	initX := []float64{-3.0, 4.0}
	actual := GradientDescent(function2, initX, 0.1, 100)
	expected := Vector{-6.111107928998789e-10, 8.148143905314271e-10}
	if actual.NotEqual(expected) {
		log.Println(actual, expected)
		t.Fail()
	}
}

func TestGradientDescent2(t *testing.T) {
	initX := []float64{-3.0, 4.0}
	actual := GradientDescent(function2, initX, 10.0, 100)
	expected := Vector{-2.5898374737328363e+13, -1.2952486168965398e+12}
	if actual.NotEqual(expected) {
		log.Println(actual, expected)
		t.Fail()
	}
}

func TestGradientDescent3(t *testing.T) {
	initX := []float64{-3.0, 4.0}
	actual := GradientDescent(function2, initX, 1e-10, 100)
	expected := Vector{-2.999999939999995, 3.9999999199999934}
	if actual.NotEqual(expected) {
		log.Println(actual, expected)
		t.Fail()
	}
}
