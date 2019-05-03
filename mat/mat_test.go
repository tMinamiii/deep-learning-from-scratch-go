package mat

import (
	"testing"

	"github.com/naronA/zero_deeplearning/vec"
)

func TestMul_1(t *testing.T) {
	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
	m2, _ := NewMatrix(2, 1, vec.Vector{3, 4})
	actual := m1.Mul(m2)
	expected, _ := NewMatrix(1, 1, vec.Vector{11})
	if !actual.Equal(expected) {
		t.Fail()
	}
}

func TestMul_2(t *testing.T) {
	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
	m2, _ := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	actual := m1.Mul(m2)
	expected, _ := NewMatrix(1, 2, vec.Vector{
		7, 10,
	})
	if !actual.Equal(expected) {
		t.Fail()
	}
}

func TestMul_3(t *testing.T) {
	m1, _ := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	m2, _ := NewMatrix(2, 2, vec.Vector{
		1, 0,
		0, 1,
	})
	actual := m1.Mul(m2)
	expected, _ := NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	if !actual.Equal(expected) {
		t.Fail()
	}

}

func TestMul_4(t *testing.T) {
	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
	m2, _ := NewMatrix(2, 3, vec.Vector{
		1, 1, 1,
		2, 2, 2,
	})
	actual := m1.Mul(m2)
	expected, _ := NewMatrix(1, 3, vec.Vector{5, 5, 5})
	if !actual.Equal(expected) {
		t.Fail()
	}
}

func TestAdd_1(t *testing.T) {
	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
	m2, _ := NewMatrix(1, 2, vec.Vector{3, 4})
	actual := m1.Add(m2)
	expected, _ := NewMatrix(1, 2, vec.Vector{4, 6})
	if !actual.Equal(expected) {
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
	if !actual.Equal(expected) {
		t.Fail()
	}
}
