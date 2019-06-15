package tensor

type Tensor6D []Tensor5D

func (t Tensor6D) Shape() (int, int, int, int, int, int) {
	A := len(t)
	B, N, C, H, W := t[0].Shape()
	return A, B, N, C, H, W
}

func (t Tensor6D) element(a, b, n, c, h, w int) float64 {
	return t[a].element(b, n, c, h, w)
}

func (t Tensor6D) assign(value float64, a, b, n, c, h, w int) {
	t[a].assign(value, b, n, c, h, w)
}

func zerosT6D(n, c, fh, fw, oh, ow int) Tensor6D {
	t6d := make(Tensor6D, n)
	for i := range t6d {
		t6d[i] = zerosT5D(c, fh, fw, oh, ow)
	}
	return t6d
}

func (t Tensor6D) window(x, y, h, w int) Tensor6D {
	newT6D := make(Tensor6D, len(t))
	for i, t5d := range t {
		newT6D[i] = t5d.window(x, y, h, w)
	}
	return newT6D
}

func (t Tensor6D) transpose(a, b, c, d, e, f int) Tensor6D {
	u, v, w, x, y, z := t.Shape()
	shape := []int{u, v, w, x, y, z}
	t6d := zerosT6D(shape[a], shape[b], shape[c], shape[d], shape[e], shape[f])
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
							v := t.element(i, j, k, l, n, m)
							t6d.assign(v, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5])
						}
					}
				}
			}
		}
	}
	return t6d
}

func (t Tensor6D) pad(size int) Tensor6D {
	newT6D := make(Tensor6D, len(t))
	for i, t5d := range t {
		padded := t5d.pad(size)
		newT6D[i] = padded
	}
	return newT6D
}

func (t Tensor6D) equal(x Tensor6D) bool {
	for i := range t {
		if !t[i].equal(x[i]) {
			return false
		}
	}
	return true
}

func (t Tensor6D) sliceTo4D(x, y int) Tensor4D {
	t6d := t
	t4d := make(Tensor4D, 0, len(t6d))
	for _, ncolT5d := range t6d {
		t3d := make(Tensor3D, 0, len(ncolT5d))
		for _, ncolT4d := range ncolT5d {
			ncolMat := ncolT4d[x][y]
			t3d = append(t3d, ncolMat)
		}
		t4d = append(t4d, t3d)
	}
	return t4d
}
