package num

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/vec"
)

func TestDot_1(t *testing.T) {
	m1 := Matrix{vec.Vector{1, 2}}
	m2 := Matrix{vec.Vector{3}, vec.Vector{4}}
	actual := Dot(m1, m2)
	expected := Matrix{vec.Vector{11}}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestDot_2(t *testing.T) {
	m1 := Matrix{vec.Vector{1, 2}}
	m2 := Matrix{vec.Vector{1, 2}, vec.Vector{3, 4}}
	actual := Dot(m1, m2)
	expected := Matrix{vec.Vector{7, 10}}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestDot_3(t *testing.T) {
	m1 := Matrix{vec.Vector{1, 2}, vec.Vector{3, 4}}
	m2 := Matrix{vec.Vector{1, 0}, vec.Vector{0, 1}}
	actual := Dot(m1, m2)
	expected := Matrix{vec.Vector{1, 2}, vec.Vector{3, 4}}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}

}

func TestDot_4(t *testing.T) {
	m1 := Matrix{vec.Vector{1, 2}}
	m2 := Matrix{vec.Vector{1, 1, 1}, vec.Vector{2, 2, 2}}
	actual := Dot(m1, m2)
	expected := Matrix{vec.Vector{5, 5, 5}}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestAdd_1(t *testing.T) {
	m1 := Matrix{vec.Vector{1, 2}}
	m2 := Matrix{vec.Vector{3, 4}}
	actual := Add(m1, m2)
	expected := Matrix{vec.Vector{4, 6}}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestAdd_2(t *testing.T) {
	m1 := Matrix{
		vec.Vector{1, 2},
		vec.Vector{3, 4},
	}
	m2 := Matrix{
		vec.Vector{3, 4},
		vec.Vector{4, 5},
	}
	actual := Add(m1, m2)
	expected := Matrix{vec.Vector{4, 6}, vec.Vector{7, 9}}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestT_1(t *testing.T) {
	m := Matrix{
		vec.Vector{1, 1, 1},
		vec.Vector{2, 2, 2},
	}
	actual := m.T()
	expected := Matrix{
		vec.Vector{1, 2},
		vec.Vector{1, 2},
		vec.Vector{1, 2},
	}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestT_2(t *testing.T) {
	m := Matrix{
		vec.Vector{1, 2, 3},
		vec.Vector{4, 5, 6},
		vec.Vector{7, 8, 9},
	}
	actual := m.T()
	expected := Matrix{
		vec.Vector{1, 4, 7},
		vec.Vector{2, 5, 8},
		vec.Vector{3, 6, 9},
	}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestSum_1(t *testing.T) {
	m := Matrix{
		vec.Vector{1, 2, 3},
		vec.Vector{4, 5, 6},
	}
	actual := Sum(m, 0)
	expected := Matrix{vec.Vector{5, 7, 9}}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestSum_2(t *testing.T) {
	m := Matrix{
		vec.Vector{1, 2, 3},
		vec.Vector{4, 5, 6},
		vec.Vector{7, 8, 9},
	}
	actual := Sum(m, 0)
	expected := Matrix{vec.Vector{12, 15, 18}}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestSoftmax(t *testing.T) {
	m := Matrix{
		vec.Vector{3, 1, 0},
		vec.Vector{1, 4, 0},
	}
	actual := Softmax(m)
	expected := Matrix{
		vec.Vector{0.8437947344813395, 0.11419519938459449, 0.04201006613406605},
		vec.Vector{0.0466126225779739, 0.9362395518765058, 0.01714782554552039},
	}
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}

func TestCrossEntropyError1(t *testing.T) {
	y := Matrix{
		vec.Vector{1, 0},
		vec.Vector{0, 1},
	}
	actual := CrossEntropyError(y, y)
	expected := -9.999999505838704e-08
	if actual != expected {
		fmt.Println(actual, expected)
		t.Fail()
	}
}
