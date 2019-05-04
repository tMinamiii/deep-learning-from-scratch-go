package mat

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/vec"
)

func TestDot_1(t *testing.T) {
	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
	m2, _ := NewMatrix(2, 1, vec.Vector{3, 4})
	actual := Dot(m1, m2)
	expected, _ := NewMatrix(1, 1, vec.Vector{11})
	if NotEqual(actual, expected) {
		t.Fail()
	}
}

func TestDot_2(t *testing.T) {
	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
	m2, _ := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	actual := Dot(m1, m2)
	expected, _ := NewMatrix(1, 2, vec.Vector{
		7, 10,
	})
	if NotEqual(actual, expected) {
		t.Fail()
	}
}

func TestDot_3(t *testing.T) {
	m1, _ := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	m2, _ := NewMatrix(2, 2, vec.Vector{
		1, 0,
		0, 1,
	})
	actual := Dot(m1, m2)
	expected, _ := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	if NotEqual(actual, expected) {
		t.Fail()
	}

}

func TestDot_4(t *testing.T) {
	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
	m2, _ := NewMatrix(2, 3, vec.Vector{
		1, 1, 1,
		2, 2, 2,
	})
	actual := Dot(m1, m2)
	expected, _ := NewMatrix(1, 3, vec.Vector{5, 5, 5})
	if NotEqual(actual, expected) {
		t.Fail()
	}
}

func TestAdd_1(t *testing.T) {
	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
	m2, _ := NewMatrix(1, 2, vec.Vector{3, 4})
	actual := m1.Add(m2)
	expected, _ := NewMatrix(1, 2, vec.Vector{4, 6})
	if NotEqual(actual, expected) {
		t.Fail()
	}
}

func TestAdd_2(t *testing.T) {
	m1, _ := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	m2, _ := NewMatrix(2, 2, vec.Vector{
		3, 4,
		4, 5,
	})
	actual := m1.Add(m2)
	expected, _ := NewMatrix(2, 2, vec.Vector{
		4, 6,
		7, 9,
	})
	if NotEqual(actual, expected) {
		t.Fail()
	}
}

func TestSoftmax(t *testing.T) {
	m, _ := NewMatrix(2, 3, vec.Vector{
		3, 1, 0,
		1, 4, 0,
	})
	actual := Softmax(m)
	expected, _ := NewMatrix(2, 3, vec.Vector{
		0.24458689267500713, 0.03310123639613508, 0.01217726434749398,
		0.03310123639613508, 0.6648561058377347, 0.01217726434749398,
	})
	if NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}
