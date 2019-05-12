package main

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

func main() {
	t4d := make(mat.Tensor4D, 2)
	t3d1 := make(mat.Tensor3D, 2)
	t3d1[0] = &mat.Matrix{Vector: vec.Vector{4, 9, 3, 6}, Rows: 2, Columns: 2}
	t3d1[1] = &mat.Matrix{Vector: vec.Vector{7, 9, 0, 9}, Rows: 2, Columns: 2}

	t3d2 := make(mat.Tensor3D, 2)
	t3d2[0] = &mat.Matrix{Vector: vec.Vector{4, 7, 3, 9}, Rows: 2, Columns: 2}
	t3d2[1] = &mat.Matrix{Vector: vec.Vector{4, 4, 1, 9}, Rows: 2, Columns: 2}
	t4d[0] = t3d1
	t4d[1] = t3d2
	// pad := t4d.Pad(1)

	// expected := Im2col(t4d, 2, 2, 2, 1)
	actual := make(mat.Tensor3D, 8)
	actual[0] = &mat.Matrix{Vector: vec.Vector{0, 0, 0, 4, 0, 0, 0, 7}, Rows: 1, Columns: 8}
	actual[1] = &mat.Matrix{Vector: vec.Vector{0, 0, 9, 0, 0, 0, 9, 0}, Rows: 1, Columns: 8}
	actual[2] = &mat.Matrix{Vector: vec.Vector{0, 3, 0, 0, 0, 0, 0, 0}, Rows: 1, Columns: 8}
	actual[3] = &mat.Matrix{Vector: vec.Vector{6, 0, 0, 0, 9, 0, 0, 0}, Rows: 1, Columns: 8}
	actual[4] = &mat.Matrix{Vector: vec.Vector{0, 0, 0, 4, 0, 0, 0, 4}, Rows: 1, Columns: 8}
	actual[5] = &mat.Matrix{Vector: vec.Vector{0, 0, 7, 0, 0, 0, 4, 0}, Rows: 1, Columns: 8}
	actual[6] = &mat.Matrix{Vector: vec.Vector{0, 3, 0, 0, 0, 1, 0, 0}, Rows: 1, Columns: 8}
	actual[7] = &mat.Matrix{Vector: vec.Vector{9, 0, 0, 0, 9, 0, 0, 0}, Rows: 1, Columns: 8}
	// win := pad.Window(0, 0, 2, 2)
	// fmt.Println(t3d1[0].ToCol())
	m := t4d.Im2Col(2, 2, 2, 1)
	// fmt.Println(m)
	fmt.Println(m.ReshapeTo4D(2, 2, 2, -1)[0][1])
	// fmt.Println(m.ReshapeTo4D(2, 2, 2, -1).ReshapeToMat(1, -1).ReshapeTo4D(2, 8, 2, 2))
	// fmt.Println(win)
}
