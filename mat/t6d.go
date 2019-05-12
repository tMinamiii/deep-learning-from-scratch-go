package mat

type Tensor6D []Tensor5D

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
