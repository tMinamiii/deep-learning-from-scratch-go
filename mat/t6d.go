package mat

import "fmt"

type Tensor6D []Tensor5D

func (t Tensor6D) Shape() (int, int, int, int, int, int) {
	A := len(t)
	B := len(t[0])
	N := len(t[0][0])
	C := t[0][0][0].Channels()
	H, W := t[0][0][0][0].Shape()
	return A, B, N, C, H, W
}

func (t Tensor6D) Element(a, b, n, c, h, w int) float64 {
	return t[a].Element(b, n, c, h, w)
}

func (t Tensor6D) Assign(value float64, a, b, n, c, h, w int) {
	t[a].Assign(value, b, n, c, h, w)
}

func (t Tensor6D) Pad(size int) Tensor6D {
	newT6D := Tensor6D{}
	for _, t4d := range t {
		padded := t4d.Pad(size)
		newT6D = append(newT6D, padded)
	}
	return newT6D
}

func ZerosT6D(n, c, fh, fw, oh, ow int) Tensor6D {
	t5d := ZerosT5D(c, fh, fw, oh, ow)
	t6d := make(Tensor6D, n)
	for i := range t6d {
		t6d[i] = t5d
	}
	return t6d
}

func (t Tensor6D) Transpose(a, b, c, d, e, f int) Tensor6D {
	u, v, w, x, y, z := t.Shape()
	shape := []int{u, v, w, x, y, z}
	t6d := ZerosT6D(shape[a], shape[b], shape[c], shape[d], shape[e], shape[f])
	fmt.Println(t)
	for i, et5d := range t {
		for j, et4d := range et5d {
			for k, et3d := range et4d {
				for l, emat := range et3d {
					for n := 0; n < emat.Rows; n++ {
						for m := 0; m < emat.Columns; m++ {
							oldIdx := []int{i, j, k, l, n, m}
							idx := make([]int, 6)
							idx[0] = oldIdx[a]
							idx[1] = oldIdx[b]
							idx[2] = oldIdx[c]
							idx[3] = oldIdx[d]
							idx[4] = oldIdx[e]
							idx[5] = oldIdx[f]
							// fmt.Println(i, j, k, l)
							// fmt.Println(" ", idx[0], idx[1], idx[2], idx[3])
							v := t.Element(i, j, k, l, n, m)
							t6d.Assign(v, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5])
						}
					}
				}
			}
		}
	}
	return t6d
}
