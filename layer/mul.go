package layer

type MulLayer struct {
	X float64
	Y float64
}

func (ml *MulLayer) forward(x, y float64) float64 {
	ml.X = x
	ml.Y = y
	out := x * y
	return out
}

func (ml *MulLayer) backward(dout float64) (float64, float64) {
	dx := dout * ml.Y
	dy := dout * ml.X
	return dx, dy
}
