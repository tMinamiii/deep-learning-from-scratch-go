package num

import (
	"testing"

	"github.com/naronA/zero_deeplearning/vec"
)

func testCoversion(t *testing.T) {
	t4d := make(Tensor4D, 2)
	t3d1 := make(Tensor3D, 2)
	t3d1[0] = &Matrix{Vector: vec.Vector{4, 9, 3, 6}, Rows: 2, Columns: 2}
	t3d1[1] = &Matrix{Vector: vec.Vector{7, 9, 0, 9}, Rows: 2, Columns: 2}

	t3d2 := make(Tensor3D, 2)
	t3d2[0] = &Matrix{Vector: vec.Vector{4, 7, 3, 9}, Rows: 2, Columns: 2}
	t3d2[1] = &Matrix{Vector: vec.Vector{4, 4, 1, 9}, Rows: 2, Columns: 2}
	t4d[0] = t3d1
	t4d[1] = t3d2
	// pad := t4d.Pad(1)

	// expected := Im2col(t4d, 2, 2, 2, 1)
	expected := &Matrix{Vector: vec.Vector{
		0, 0, 0, 4, 0, 0, 0, 7,
		0, 0, 9, 0, 0, 0, 9, 0,
		0, 3, 0, 0, 0, 0, 0, 0,
		6, 0, 0, 0, 9, 0, 0, 0,
		0, 0, 0, 4, 0, 0, 0, 4,
		0, 0, 7, 0, 0, 0, 4, 0,
		0, 3, 0, 0, 0, 1, 0, 0,
		9, 0, 0, 0, 9, 0, 0, 0,
	}, Rows: 8, Columns: 8}

	actual := t4d.Im2Col(2, 2, 2, 1)
	if NotEqual(expected, actual) {
		t.Fail()
	}

	img := actual.Col2Img([]int{2, 2, 2, 2}, 2, 2, 2, 1)
	if !EqualT4D(t4d, img) {
		t.Fail()
	}
}
