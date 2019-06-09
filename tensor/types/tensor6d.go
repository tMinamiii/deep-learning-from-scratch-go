package types

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

func ZerosT6D(n, c, fh, fw, oh, ow int) Tensor6D {
	t6d := make(Tensor6D, n)
	for i := range t6d {
		t6d[i] = ZerosT5D(c, fh, fw, oh, ow)
	}
	return t6d
}
