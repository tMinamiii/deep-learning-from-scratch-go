package mat

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/vec"
)

type Tensor4D []Tensor3D

func (t Tensor4D) Im2Col(fw, fh, stride, pad int) *Matrix {
	nv := vec.Vector{}
	for _, e := range t {
		nv = append(nv, e.Im2Col(fw, fh, stride, pad).Vector...)
	}
	N, C, H, _ := t.Shape()
	return &Matrix{
		Vector:  nv,
		Rows:    N * C * H,
		Columns: fw * fh * C,
	}
}

func (t Tensor4D) Size() int {
	n, c, h, w := t.Shape()
	return n * c * h * w
}

func (t Tensor4D) Flatten() vec.Vector {
	v := vec.Vector{}
	for _, e := range t {
		v = append(v, e.Flatten()...)
	}
	return v
}

func (t Tensor4D) Transpose(a, b, c, d int) Tensor4D {
	// n0 := t[:]
	// n1 := t[:][:]
	// n2 := t[:][:][:]
	// n3 := t[:][:][:][:]
	return nil
}

func (t Tensor4D) ReshapeToMat(row, col int) *Matrix {
	size := t.Size()
	if col == -1 {
		col = int(size / row)
	}
	flat := t.Flatten()
	return &Matrix{
		Vector:  flat,
		Rows:    row,
		Columns: col,
	}
}

func (t Tensor4D) Window(x, y, h, w int) Tensor4D {
	newT4D := Tensor4D{}
	for _, mat := range t {
		newT4D = append(newT4D, mat.Window(x, y, h, w))
	}
	return newT4D
}

func (t Tensor4D) Pad(size int) Tensor4D {
	newT4D := Tensor4D{}
	for _, t3d := range t {
		padded := t3d.Pad(size)
		newT4D = append(newT4D, padded)
	}
	return newT4D
}

func (t Tensor4D) Shape() (int, int, int, int) {
	N := len(t)
	C := t[0].Channels()
	H, W := t[0][0].Shape()
	return N, C, H, W
}

func ZerosT4D(n, c, h, w int) Tensor4D {
	t3d := ZerosT3D(c, h, w)
	t4d := make(Tensor4D, c)
	for i := range t4d {
		t4d[i] = t3d
	}
	return t4d
}

/*
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
*/
func Im2col(input Tensor4D, fh, fw, stride, pad int) *Matrix {
	N, C, H, W := input.Shape()
	oh := (H+2*pad-fh)/stride + 1
	ow := (W+2*pad-fw)/stride + 1
	img := input.Pad(pad) // 4D
	col := ZerosT6D(N, C, fh, fw, oh, ow)
	for i := 0; i < N; i++ {
		for j := 0; j < C; j++ {
			for k := 0; k < H; k += stride {
				for l := 0; l < W; l += stride {
					window := img.Window(k, l, fw, fw) // 4D
					fmt.Println(col, window)
				}
			}
		}
	}
	return nil
}
