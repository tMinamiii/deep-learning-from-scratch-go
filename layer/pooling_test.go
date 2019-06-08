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
			num.NewMatrix(
				vec.Vector{
					4, 9, 0, 1,
					3, 6, 4, 5,
					0, 7, 2, 4,
					6, 5, 9, 2,
				}, 4, 4,
			),
			num.NewMatrix(
				vec.Vector{
					6, 8, 1, 2,
					4, 1, 8, 1,
					1, 0, 4, 3,
					2, 6, 4, 0,
				}, 4, 4,
			),
			num.NewMatrix(
				vec.Vector{
					3, 9, 0, 1,
					1, 5, 0, 4,
					4, 7, 3, 2,
					5, 7, 6, 5,
				}, 4, 4,
			),
		},
	}

}
func TestPooling(t *testing.T) {
	t4d := SampleT4d()
	pool := NewPooling(2, 2, 1, 0)
	actual := pool.Forward(t4d).(num.Tensor4D)
	expected := num.Tensor4D{
		num.Tensor3D{
			num.NewMatrix(
				vec.Vector{
					9, 9, 5,
					7, 7, 5,
					7, 9, 9,
				}, 3, 3,
			),
			num.NewMatrix(
				vec.Vector{
					8, 8, 8,
					4, 8, 8,
					6, 6, 4,
				}, 3, 3,
			),
			num.NewMatrix(
				vec.Vector{
					9, 9, 4,
					7, 7, 4,
					7, 7, 6,
				}, 3, 3,
			),
		},
	}
	if !num.EqualT4D(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}

	actual2 := pool.Backward(expected).(num.Tensor4D)
	expected2 := num.Tensor4D{
		num.Tensor3D{
			num.NewMatrix(
				vec.Vector{
					0, 18, 0, 0,
					0, 0, 0, 10,
					0, 21, 0, 0,
					0, 0, 18, 0,
				}, 4, 4,
			),
			num.NewMatrix(
				vec.Vector{
					0, 16, 0, 0,
					4, 0, 24, 0,
					0, 0, 4, 0,
					0, 12, 0, 0,
				},
				4,
				4,
			),
			num.NewMatrix(
				vec.Vector{
					0, 18, 0, 0,
					0, 0, 0, 8,
					0, 28, 0, 0,
					0, 0, 6, 0,
				}, 4, 4,
			),
		},
	}
	if !num.EqualT4D(actual2, expected2) {
		fmt.Println(actual2, expected2)
		t.Fail()
	}
}
