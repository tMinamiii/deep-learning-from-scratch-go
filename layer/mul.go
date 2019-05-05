package layer

type MulLayer struct {
	X float64
	Y float64
}

func (self *MulLayer) forward(x, y float64) float64 {
	self.X = x
	self.Y = y
	out := x * y
	return out
}

func (self *MulLayer) backward(dout float64) (float64, float64) {
	dx := dout * self.Y
	dy := dout * self.X
	return dx, dy
}
