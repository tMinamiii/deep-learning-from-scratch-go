package layer

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

func SampleT4d() num.Tensor4D {
	return num.Tensor4D{
		num.Tensor3D{
			&num.Matrix{
				Vector: vec.Vector{
					4, 9, 0, 1,
					3, 6, 4, 5,
					0, 7, 2, 4,
					6, 5, 9, 2,
				},
				Rows:    4,
				Columns: 4,
			},
			&num.Matrix{
				Vector: vec.Vector{
					6, 8, 1, 2,
					4, 1, 8, 1,
					1, 0, 4, 3,
					2, 6, 4, 0,
				},
				Rows:    4,
				Columns: 4,
			},
			&num.Matrix{
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

}
func TestPooling(t *testing.T) {
	t4d := SampleT4d()
	pool := NewPooling(2, 2, 1, 0)
	actual := pool.Forward(t4d)
	expected := num.Tensor4D{
		num.Tensor3D{
			&num.Matrix{
				Vector: vec.Vector{
					9, 9, 5,
					7, 7, 5,
					7, 9, 9,
				},
				Rows:    3,
				Columns: 3,
			},
			&num.Matrix{
				Vector: vec.Vector{
					8, 8, 8,
					4, 8, 8,
					6, 6, 4,
				},
				Rows:    3,
				Columns: 3,
			},
			&num.Matrix{
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
	if !num.EqualT4D(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}

	actual2 := pool.Backward(expected)
	expected2 := num.Tensor4D{
		num.Tensor3D{
			&num.Matrix{
				Vector: vec.Vector{
					0, 18, 0, 0,
					0, 0, 0, 10,
					0, 21, 0, 0,
					0, 0, 18, 0},
				Rows:    4,
				Columns: 4,
			},

			&num.Matrix{
				Vector: vec.Vector{
					0, 16, 0, 0,
					4, 0, 24, 0,
					0, 0, 4, 0,
					0, 12, 0, 0,
				},
				Rows:    4,
				Columns: 4,
			},
			&num.Matrix{
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
	if !num.EqualT4D(actual2, expected2) {
		fmt.Println(actual2, expected2)
		t.Fail()
	}
}
