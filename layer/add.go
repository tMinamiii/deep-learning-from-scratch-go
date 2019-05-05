package layer

type AddLayer struct {
}

func (ad *AddLayer) forward(x, y float64) float64 {
	out := x + y
	return out
}

func (ad *AddLayer) backward(dout float64) (float64, float64) {
	dx := dout * 1
	dy := dout * 1
	return dx, dy
}
