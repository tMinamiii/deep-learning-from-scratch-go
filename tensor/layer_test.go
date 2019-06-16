package tensor

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/vec"
)

func SampleT4d() *Tensor {
	t4d := Tensor4D{
		Tensor3D{
			&Matrix{
				Vector: vec.Vector{
					4, 9, 0, 1,
					3, 6, 4, 5,
					0, 7, 2, 4,
					6, 5, 9, 2,
				},
				Rows:    4,
				Columns: 4,
			},
			&Matrix{
				Vector: vec.Vector{
					6, 8, 1, 2,
					4, 1, 8, 1,
					1, 0, 4, 3,
					2, 6, 4, 0,
				},
				Rows:    4,
				Columns: 4,
			},
			&Matrix{
				Vector: vec.Vector{
					3, 9, 0, 1,
					1, 5, 0, 4,
					4, 7, 3, 2,
					5, 7, 6, 5,
				},
				Rows:    4,
				Columns: 4,
			},
		},
	}
	return &Tensor{T4D: t4d, Shape: []int{1, 3, 4, 4}}

}
func TestPooling(t *testing.T) {
	t4d := SampleT4d()
	pool := NewPooling(2, 2, 1, 0)
	actual := pool.Forward(t4d)
	expt4d := Tensor4D{
		Tensor3D{
			&Matrix{
				Vector: vec.Vector{
					9, 9, 5,
					7, 7, 5,
					7, 9, 9,
				},
				Rows:    3,
				Columns: 3,
			},
			&Matrix{
				Vector: vec.Vector{
					8, 8, 8,
					4, 8, 8,
					6, 6, 4,
				},
				Rows:    3,
				Columns: 3,
			},
			&Matrix{
				Vector: vec.Vector{
					9, 9, 4,
					7, 7, 4,
					7, 7, 6,
				},
				Rows:    3,
				Columns: 3,
			},
		},
	}
	expected := &Tensor{T4D: expt4d, Shape: []int{1, 3, 3, 3}}
	if actual.NotEqual(expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}

	actual2 := pool.Backward(expected)
	expt4d2 := Tensor4D{
		Tensor3D{
			&Matrix{
				Vector: vec.Vector{
					0, 18, 0, 0,
					0, 0, 0, 10,
					0, 21, 0, 0,
					0, 0, 18, 0},
				Rows:    4,
				Columns: 4,
			},
			&Matrix{
				Vector: vec.Vector{
					0, 16, 0, 0,
					4, 0, 24, 0,
					0, 0, 4, 0,
					0, 12, 0, 0,
				},
				Rows:    4,
				Columns: 4,
			},
			&Matrix{
				Vector: vec.Vector{
					0, 18, 0, 0,
					0, 0, 0, 8,
					0, 28, 0, 0,
					0, 0, 6, 0,
				},
				Rows:    4,
				Columns: 4,
			},
		},
	}
	expected2 := &Tensor{T4D: expt4d2, Shape: []int{1, 3, 4, 4}}
	if actual2.NotEqual(expected2) {
		fmt.Println(actual2, expected2)
		t.Fail()
	}
}
