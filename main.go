package main

import (
    "fmt"

    "github.com/naronA/zero_deeplearning/layer"
    "github.com/naronA/zero_deeplearning/num"
    "github.com/naronA/zero_deeplearning/vec"
)

func submain() {
    t4d := make(num.Tensor4D, 2)
    t3d1 := make(num.Tensor3D, 2)
    t3d1[0] = &num.Matrix{Vector: vec.Vector{4, 9, 3, 6}, Rows: 2, Columns: 2}
    t3d1[1] = &num.Matrix{Vector: vec.Vector{7, 9, 0, 9}, Rows: 2, Columns: 2}

    t3d2 := make(num.Tensor3D, 2)
    t3d2[0] = &num.Matrix{Vector: vec.Vector{4, 7, 3, 9}, Rows: 2, Columns: 2}
    t3d2[1] = &num.Matrix{Vector: vec.Vector{4, 4, 1, 9}, Rows: 2, Columns: 2}
    t4d[0] = t3d1
    t4d[1] = t3d2
    // pad := t4d.Pad(1)

    // expected := Im2col(t4d, 2, 2, 2, 1)
    actual := make(num.Tensor3D, 8)
    actual[0] = &num.Matrix{Vector: vec.Vector{0, 0, 0, 4, 0, 0, 0, 7}, Rows: 1, Columns: 8}
    actual[1] = &num.Matrix{Vector: vec.Vector{0, 0, 9, 0, 0, 0, 9, 0}, Rows: 1, Columns: 8}
    actual[2] = &num.Matrix{Vector: vec.Vector{0, 3, 0, 0, 0, 0, 0, 0}, Rows: 1, Columns: 8}
    actual[3] = &num.Matrix{Vector: vec.Vector{6, 0, 0, 0, 9, 0, 0, 0}, Rows: 1, Columns: 8}
    actual[4] = &num.Matrix{Vector: vec.Vector{0, 0, 0, 4, 0, 0, 0, 4}, Rows: 1, Columns: 8}
    actual[5] = &num.Matrix{Vector: vec.Vector{0, 0, 7, 0, 0, 0, 4, 0}, Rows: 1, Columns: 8}
    actual[6] = &num.Matrix{Vector: vec.Vector{0, 3, 0, 0, 0, 1, 0, 0}, Rows: 1, Columns: 8}
    actual[7] = &num.Matrix{Vector: vec.Vector{9, 0, 0, 0, 9, 0, 0, 0}, Rows: 1, Columns: 8}
    // win := pad.Window(0, 0, 2, 2)
    // fmt.Println(t3d1[0].ToCol())
    m := t4d.Im2Col(2, 2, 2, 1)
    // fmt.Println(m)
    img := m.Col2Img([]int{2, 2, 2, 2}, 2, 2, 2, 1)

    fmt.Println(img)
    // fmt.Println(m.ReshapeTo4D(2, 2, 2, -1)[0][0].RowVecs())
    // res4d := m.ReshapeTo4D(2, 2, 2, -1)
    // fmt.Println(res4d.Transpose(3, 0, 2, 1))
    // fmt.Println(m.ReshapeTo4D(2, 2, 2, -1)[0][0].RowVecs())
    // fmt.Println(m.ReshapeTo4D(2, 2, 2, -1).ReshapeToMat(1, -1).ReshapeTo4D(2, 8, 2, 2))
    // fmt.Println(win)
}

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

func main() {
    t4d := SampleT4d()
    pool := layer.NewPooling(2, 2, 1, 0)
    out := pool.Forward(t4d)
	pool.Backward(out)
}
