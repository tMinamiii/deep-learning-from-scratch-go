package mat

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
	t4d := ZerosT4D(b, c, h, w)
	t5d := make(Tensor5D, a)
	for i := range t5d {
		t5d[i] = t4d
	}
	return t5d
}
