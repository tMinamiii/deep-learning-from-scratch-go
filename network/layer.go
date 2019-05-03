package network

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

type AddLayer struct {
}

func (al *AddLayer) forward(x, y float64) float64{
	out := x + y
	return out
}

func (al *AddLayer) backward(dout float64) (float64, float64) {
	dx := dout * 1
	dy := dout * 1
	return dx, dy
}

type Relu struct {
	Mask bool
}

func (r *Relu) forward(x float64) {
	r.Mask = x <= 0
	out := x
	out[Mask] = 0
	return out
}

