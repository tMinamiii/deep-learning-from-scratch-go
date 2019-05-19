package mat

import (
	"github.com/naronA/zero_deeplearning/vec"
)

// func (t Tensor3D) Im2Col(fw, fh, stride, pad int) *Matrix {
// 	nV := vec.Vector{}
// 	count := 0
// 	for x := 0; x <= t[0].Columns-fw+2*pad; x += stride {
// 		for y := 0; y <= t[0].Rows-fh+2*pad; y += stride {
// 			count++
// 			for _, e := range t {
// 				padE := e.Pad(pad)
// 				nV = append(nV, padE.Window(x, y, fw, fh).Vector...)
// 			}
// 		}
// 	}
// 	return &Matrix{
// 		Vector:  nV,
// 		Rows:    count,
// 		Columns: fh * fw * len(t),
// 	}
// }

func (t Tensor4D) Im2Col(fw, fh, stride, pad int) *Matrix {
	colVec := vec.Vector{}
	for _, t3d := range t {
		nV := vec.Vector{}
		for x := 0; x <= t3d[0].Columns-fw+2*pad; x += stride {
			for y := 0; y <= t3d[0].Rows-fh+2*pad; y += stride {
				for _, ma := range t3d {
					padE := ma.Pad(pad)
					nV = append(nV, padE.Window(x, y, fw, fh).Vector...)
				}
			}
		}
		colVec = append(colVec, nV...)
	}

	N, C, H, _ := t.Shape()
	return &Matrix{
		Vector:  colVec,
		Rows:    N * C * H,
		Columns: fw * fh * C,
	}
}

type Tensor4DIndex struct {
	N int
	C int
	H int
	W int
}

type Tensor4DSlice struct {
	Actual   Tensor4D
	Indices  []*Tensor4DIndex
	NewShape []int
}

func AddAssign(t1 *Tensor4DSlice, t2 Tensor4D) {
	t2flat := t2.Flatten()
	for i, idx := range t1.Indices {
		add := t1.Actual[idx.N][idx.C].Element(idx.H, idx.W) + t2flat[i]
		t1.Actual[idx.N][idx.C].Assign(add, idx.H, idx.W)
	}
}

func (t Tensor6D) slice(x, y int) Tensor4D {
	t4d := Tensor4D{}
	for _, ncolT5d := range t {
		t3d := Tensor3D{}
		for _, ncolT4d := range ncolT5d {
			ncolMat := ncolT4d[x][y]
			t3d = append(t3d, ncolMat)
		}
		t4d = append(t4d, t3d)
	}
	return t4d
}

func (t Tensor4D) StrideSlice(y, yMax, x, xMax, stride int) *Tensor4DSlice {
	n, c, _, _ := t.Shape()
	indices := []*Tensor4DIndex{}
	totalRows := 0
	for i := y; i < yMax; i += stride {
		totalRows++
	}
	totalColumns := 0
	for j := x; j < xMax; j += stride {
		totalColumns++
	}
	for n, imgT3D := range t {
		for c := range imgT3D {
			for i := y; i < yMax; i += stride {
				for j := x; j < xMax; j += stride {
					index := &Tensor4DIndex{n, c, i, j}
					indices = append(indices, index)
				}
			}
		}
	}
	return &Tensor4DSlice{
		Actual:   t,
		Indices:  indices,
		NewShape: []int{n, c, totalRows, totalColumns},
	}
}

/**
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
*/
func (m *Matrix) Col2Img(shape []int, fh, fw, stride, pad int) Tensor4D {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	outH := (H+2*pad-fh)/stride + 1
	outW := (W+2*pad-fw)/stride + 1
	ncol := m.ReshapeTo6D(N, outH, outW, C, fh, fw).Transpose(0, 3, 4, 5, 1, 2)
	img := ZerosT4D(N, C, H+2*pad+stride-1, W+2*pad+stride-1)
	imgMats := Tensor3D{}
	for _, imgT3D := range img {
		imgMats = append(imgMats, imgT3D...)
	}
	for y := 0; y < fh; y++ {
		yMax := y + stride*outH
		for x := 0; x < fw; x++ {
			xMax := x + stride*outW
			slice := img.StrideSlice(y, yMax, x, xMax, stride)
			ncolSlice := ncol.slice(y, x)
			AddAssign(slice, ncolSlice)
		}
	}

	return img
}
