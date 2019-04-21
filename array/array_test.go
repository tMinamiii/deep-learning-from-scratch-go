package array

import (
	"log"
	"math"
	"testing"
)

func TestArrayMulti(t *testing.T) {
	x := []float64{2, 4, 5}
	w := []float64{1, 2, 3}
	result, err := Multi(x, w)
	if err != nil {
		t.Fail()
	}
	if !Equal(result, []float64{2, 8, 15}) {
		t.Fail()
	}
	if Equal(result, []float64{1, 4, 10}) {
		t.Fail()
	}
	if Equal(result, []float64{2, 9, 15}) {
		t.Fail()
	}
	if Equal(result, []float64{2, 9}) {
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
	ary := []float64{1, 2, 3, 4, 5}
	result := Divide(ary, 10)
	expected := []float64{1.0 / 10.0, 2.0 / 10.0, 3.0 / 10.0, 4.0 / 10.0, 5.0 / 10.0}
	if !Equal(result, expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestArraySub(t *testing.T) {
	ary := []float64{1, 2, 3, 4, 5}
	result := Sub(ary, 2)
	expected := []float64{1 - 2, 2 - 2, 3 - 2, 4 - 2, 5 - 2}
	if !Equal(result, expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestArrayExp(t *testing.T) {
	ary := []float64{1, 2, 3, 4, 5}
	result := Exp(ary)
	expected := []float64{math.Exp(1.0), math.Exp(2.0), math.Exp(3.0), math.Exp(4.0), math.Exp(5.0)}
	if !Equal(result, expected) {
		log.Println(result, expected)
		t.Fail()
	}
}

func TestSoftmax(t *testing.T) {
	actual := Softmax([]float64{-1, 0, 1, 10})
	expected := []float64{1.669860300844509e-05, 4.539150911850783e-05, 0.0001233869144031729, 0.9998145229734698}
	if !Equal(actual, expected) {
		log.Println(actual, expected)
		t.Fail()
	}
}
