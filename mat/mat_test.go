package mat

import (
	"testing"
)

func TestMul_1(t *testing.T) {
	m1, _ := NewMat64(1, 2, []float64{1, 2})
	m2, _ := NewMat64(2, 1, []float64{3, 4})
	actual := Mul(m1, m2)
	expected, _ := NewMat64(1, 1, []float64{11})
	if !Equal(actual, expected) {
		t.Fail()
	}
}

func TestMul_2(t *testing.T) {
	m1, _ := NewMat64(1, 2, []float64{1, 2})
	m2, _ := NewMat64(2, 2, []float64{
		1, 2,
		3, 4,
	})
	actual := Mul(m1, m2)
	expected, _ := NewMat64(1, 2, []float64{
		7, 10,
	})
	if !Equal(actual, expected) {
		t.Fail()
	}
}

func TestMul_3(t *testing.T) {
	m1, _ := NewMat64(2, 2, []float64{
		1, 2,
		3, 4,
	})
	m2, _ := NewMat64(2, 2, []float64{
		1, 0,
		0, 1,
	})
	actual := Mul(m1, m2)
	expected, _ := NewMat64(2, 2, []float64{
		1, 2,
		3, 4,
	})
	if !Equal(actual, expected) {
		t.Fail()
	}

}

func TestMul_4(t *testing.T) {
	m1, _ := NewMat64(1, 2, []float64{1, 2})
	m2, _ := NewMat64(2, 3, []float64{
		1, 1, 1,
		2, 2, 2,
	})
	actual := Mul(m1, m2)
	expected, _ := NewMat64(1, 3, []float64{5, 5, 5})
	if !Equal(actual, expected) {
		t.Fail()
	}
}

func TestAdd_1(t *testing.T) {
	m1, _ := NewMat64(1, 2, []float64{1, 2})
	m2, _ := NewMat64(1, 2, []float64{3, 4})
	actual := Add(m1, m2)
	expected, _ := NewMat64(1, 2, []float64{4, 6})
	if !Equal(actual, expected) {
		t.Fail()
	}
}

func TestAdd_2(t *testing.T) {
	m1, _ := NewMat64(2, 2, []float64{
		1, 2,
		3, 4,
	})
	m2, _ := NewMat64(2, 2, []float64{
		3, 4,
		4, 5,
	})
	actual := Add(m1, m2)
	expected, _ := NewMat64(2, 2, []float64{
		4, 6,
		7, 9,
	})
	if !Equal(actual, expected) {
		t.Fail()
	}
}
