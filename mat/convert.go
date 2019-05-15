package mat

import (
	"fmt"

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
	for _, ncolT5d := range ncol {
		for _, ncolT4d := range ncolT5d {
			for _, imgT3D := range img {
				for x := 0; x <= imgT3D[0].Columns-fw+2*pad; x += stride {
					for y := 0; y <= imgT3D[0].Rows-fh+2*pad; y += stride {
						for _, imgMat := range imgT3D {
							// padE := imgMat.Pad(pad)
							fmt.Println(x, y)
							ncolMat := ncolT4d[y][x]
							imgMat.AssignWindow(ncolMat, x, y, fw, fh)
						}
					}
				}
			}
		}
	}

	return img
}
