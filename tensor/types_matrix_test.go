package tensor

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/vec"
)

func TestDot_1(t *testing.T) {
	m1 := &Tensor{Mat: &Matrix{Rows: 1, Columns: 2, Vector: vec.Vector{1, 2}}, Shape: []int{1, 2}}
	m2 := &Tensor{Mat: &Matrix{Rows: 2, Columns: 1, Vector: vec.Vector{3, 4}}, Shape: []int{2, 1}}
	actual := Dot(m1, m2)
	expected := &Tensor{Mat: &Matrix{Rows: 1, Columns: 1, Vector: vec.Vector{11}}, Shape: []int{1, 1}}
	if actual.NotEqual(expected) {
		fmt.Println(actual)
		fmt.Println(expected)
		t.Fail()
	}
}

// func TestDot_2(t *testing.T) {
// 	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
// 	m2, _ := NewMatrix(2, 2, vec.Vector{
// 		1, 2,
// 		3, 4,
// 	})
// 	actual := Dot(m1, m2)
// 	expected, _ := NewMatrix(1, 2, vec.Vector{
// 		7, 10,
// 	})
// 	if NotEqual(actual, expected) {
// 		t.Fail()
// 	}
// }
//
// func TestDot_3(t *testing.T) {
// 	m1, _ := NewMatrix(2, 2, vec.Vector{
// 		1, 2,
// 		3, 4,
// 	})
// 	m2, _ := NewMatrix(2, 2, vec.Vector{
// 		1, 0,
// 		0, 1,
// 	})
// 	actual := Dot(m1, m2)
// 	expected, _ := NewMatrix(2, 2, vec.Vector{
// 		1, 2,
// 		3, 4,
// 	})
// 	if NotEqual(actual, expected) {
// 		t.Fail()
// 	}
//
// }
//
// func TestDot_4(t *testing.T) {
// 	m1, _ := NewMatrix(1, 2, vec.Vector{1, 2})
// 	m2, _ := NewMatrix(2, 3, vec.Vector{
// 		1, 1, 1,
// 		2, 2, 2,
// 	})
// 	actual := Dot(m1, m2)
// 	expected, _ := NewMatrix(1, 3, vec.Vector{5, 5, 5})
// 	if NotEqual(actual, expected) {
// 		t.Fail()
// 	}
// }
