package num

import (
	"testing"
)

func TestExp(t *testing.T) {
	if Exp(1) != 2.71828182846 {
		t.Fail()
	}
	if Exp(0) != 1 {
		t.Fail()
	}

}

func TestArrayMulti(t *testing.T) {
	x := []float64{2, 4, 5}
	w := []float64{1, 2, 3}
	result, err := ArrayMulti(x, w)
	if err != nil {
		t.Fail()
	}
	if !ArrayEqual(result, []float64{2, 8, 15}) {
		t.Fail()
	}
	if ArrayEqual(result, []float64{1, 4, 10}) {
		t.Fail()
	}
	if ArrayEqual(result, []float64{2, 9, 15}) {
		t.Fail()
	}
	if ArrayEqual(result, []float64{2, 9}) {
		t.Fail()
	}
}

func TestArraySum(t *testing.T) {
	ary := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	result := ArraySum(ary)
	if result != 55 {
		t.Fail()
	}
}
