package num

type Tensor5D []Tensor4D

func (t Tensor5D) Pad(size int) Tensor5D {
	newT5D := Tensor5D{}
	for _, t4d := range t {
		padded := t4d.Pad(size)
		newT5D = append(newT5D, padded)
	}
	return newT5D
}

func ZerosT5D(a, b, c, h, w int) Tensor5D {
	t5d := make(Tensor5D, a)
	for i := range t5d {
		t5d[i] = ZerosT4D(b, c, h, w)
	}
	return t5d
}

func (t Tensor5D) Element(b, n, c, h, w int) float64 {
	return t[b].Element(n, c, h, w)
}

func (t Tensor5D) Assign(value float64, b, n, c, h, w int) {
	t[b].Assign(value, n, c, h, w)
}
