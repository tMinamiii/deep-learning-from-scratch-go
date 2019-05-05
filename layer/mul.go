package layer

type MulLayer struct {
	X float64
	Y float64
}

func (mu *MulLayer) forward(x, y float64) float64 {
	mu.X = x
	mu.Y = y
	out := x * y
	return out
}

func (mu *MulLayer) backward(dout float64) (float64, float64) {
	dx := dout * mu.Y
	dy := dout * mu.X
	return dx, dy
}
