package array

import (
	"log"
	"math"
	"testing"
)

func TestArrayMulti(t *testing.T) {
	x := Array{2, 4, 5}
	w := Array{1, 2, 3}
	result := x.MultiArray(w)
	if result == nil {
		t.Fail()
	}
	if !result.Equal([]float64{2, 8, 15}) {
		t.Fail()
	}
	if result.Equal([]float64{1, 4, 10}) {
		t.Fail()
	}
	if result.Equal([]float64{2, 9, 15}) {
		t.Fail()
	}
	if result.Equal([]float64{2, 9}) {
		t.Fail()
	}
}

func TestArraySum(t *testing.T) {
	ary := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	result := Sum(ary)
	if result != 55 {
		t.Fail()
	}
}

func TestArrayMax(t *testing.T) {
	ary := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	result := Max(ary)
	if result != 10 {
		t.Fail()
	}
}

func TestArrayDivide(t *testing.T) {
	ary := Array{1, 2, 3, 4, 5}
	result := ary.Divide(10)
	expected := []float64{1.0 / 10.0, 2.0 / 10.0, 3.0 / 10.0, 4.0 / 10.0, 5.0 / 10.0}
	if !result.Equal(expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestArraySub(t *testing.T) {
	ary := Array{1, 2, 3, 4, 5}
	result := ary.Sub(2)
	expected := Array{1 - 2, 2 - 2, 3 - 2, 4 - 2, 5 - 2}
	if !result.Equal(expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestArrayExp(t *testing.T) {
	ary := Array{1, 2, 3, 4, 5}
	result := Exp(ary)
	expected := Array{math.Exp(1.0), math.Exp(2.0), math.Exp(3.0), math.Exp(4.0), math.Exp(5.0)}
	if !result.Equal(expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestSoftmax(t *testing.T) {
	actual := Softmax(Array{-1, 0, 1, 10})
	expected := Array{1.669860300844509e-05, 4.539150911850783e-05, 0.0001233869144031729, 0.9998145229734698}
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
