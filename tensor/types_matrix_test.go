package tensor

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/vec"
)

func TestDot1(t *testing.T) {
	m1 := NewMatrix(1, 2, vec.Vector{1, 2})
	m2 := NewMatrix(2, 1, vec.Vector{3, 4})
	actual := Dot(m1, m2)
	expected := NewMatrix(1, 1, vec.Vector{11})
	if actual.NotEqual(expected) {
		fmt.Println(actual)
		fmt.Println(expected)
		t.Fail()
	}
}

func TestDot2(t *testing.T) {
	m1 := NewMatrix(1, 2, vec.Vector{1, 2})
	m2 := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	actual := Dot(m1, m2)
	expected := NewMatrix(1, 2, vec.Vector{
		7, 10,
	})
	if actual.NotEqual(expected) {
		fmt.Println(actual)
		fmt.Println(expected)
		t.Fail()
	}
}

func TestDot3(t *testing.T) {
	m1 := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	m2 := NewMatrix(2, 2, vec.Vector{
		1, 0,
		0, 1,
	})
	actual := Dot(m1, m2)
	expected := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	if actual.NotEqual(expected) {
		t.Fail()
	}

}

func TestDot4(t *testing.T) {
	m1 := NewMatrix(1, 2, vec.Vector{1, 2})
	m2 := NewMatrix(2, 3, vec.Vector{
		1, 1, 1,
		2, 2, 2,
	})
	actual := Dot(m1, m2)
	expected := NewMatrix(1, 3, vec.Vector{5, 5, 5})
	if actual.NotEqual(expected) {
		t.Fail()
	}
}

func TestAdd1(t *testing.T) {
	m1 := NewMatrix(1, 2, vec.Vector{1, 2})
	m2 := NewMatrix(1, 2, vec.Vector{3, 4})
	actual := Add(m1, m2)
	expected := NewMatrix(1, 2, vec.Vector{4, 6})
	if actual.NotEqual(expected) {
		t.Fail()
	}
}

func TestAdd2(t *testing.T) {
	m1 := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	m2 := NewMatrix(2, 2, vec.Vector{
		3, 4,
		4, 5,
	})
	actual := Add(m1, m2)
	expected := NewMatrix(2, 2, vec.Vector{
		4, 6,
		7, 9,
	})
	if actual.NotEqual(expected) {
		t.Fail()
	}
}

func TestT_1(t *testing.T) {
	m := NewMatrix(2, 3, vec.Vector{
		1, 1, 1,
		2, 2, 2,
	})
	actual := m.T()
	expected := NewMatrix(3, 2, vec.Vector{
		1, 2,
		1, 2,
		1, 2,
	})
	if actual.NotEqual(expected) {
		t.Fail()
	}
}

func TestT_2(t *testing.T) {
	m := NewMatrix(3, 3, vec.Vector{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	actual := m.T()
	expected := NewMatrix(3, 3, vec.Vector{
		1, 4, 7,
		2, 5, 8,
		3, 6, 9,
	})
	if actual.NotEqual(expected) {
		t.Fail()
	}
}

func TestSum1(t *testing.T) {
	m := NewMatrix(2, 3, vec.Vector{
		1, 2, 3,
		4, 5, 6,
	})
	actual := m.Sum(0)
	expected := NewMatrix(1, 3, vec.Vector{5, 7, 9})
	if actual.NotEqual(expected) {
		fmt.Println(actual)
		fmt.Println(expected)
		t.Fail()
	}
}

func TestSum2(t *testing.T) {
	m := NewMatrix(3, 3, vec.Vector{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	actual := m.Sum(0)
	expected := NewMatrix(1, 3, vec.Vector{12, 15, 18})
	if actual.NotEqual(expected) {
		fmt.Println(actual)
		fmt.Println(expected)
		t.Fail()
	}
}

func TestSoftmax(t *testing.T) {
	m := NewMatrix(2, 3, vec.Vector{
		3, 1, 0,
		1, 4, 0,
	})
	actual := m.Softmax()
	expected := NewMatrix(2, 3, vec.Vector{
		0.8437947344813395, 0.11419519938459449, 0.04201006613406605,
		0.0466126225779739, 0.9362395518765058, 0.01714782554552039,
	})
	if actual.NotEqual(expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestCrossEntropyError1(t *testing.T) {
	y := NewMatrix(2, 2, vec.Vector{
		1, 0,
		0, 1,
	})
	actual := y.CrossEntropyError(y)
	expected := -9.999999505838704e-08
	if actual != expected {
		fmt.Println(actual, expected)
		t.Fail()
	}
}
