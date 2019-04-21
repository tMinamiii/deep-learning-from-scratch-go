package array

import "testing"

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
