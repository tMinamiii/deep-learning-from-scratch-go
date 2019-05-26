package num

// type Tensor []interface{}

// func (t Tensor) Flatten() vec.Vector {
// 	v := vec.Vector{}
// 	for _, e := range t {
// 		v = append(v, e.Vector...)
// 	}
// 	return v
// }

// func (t Tensor) Element(x ...int) float64 {
// 	tmp := t
// 	for i := 0; i < len(x)-2; i++ {
// 		tmp = tmp[x[i]].(Tensor)
// 	}
// 	h := x[len(x)-1]
// 	w := x[len(x)-2]
// 	return tmp.Element(h, w)
// }
//
// func (t Tensor) Assign(value float64, x ...int) {
// 	tmp := t
// 	for i := 0; i < len(x)-2; i++ {
// 		tmp = tmp[x[i]].(Tensor)
// 	}
// 	h := x[len(x)-1]
// 	w := x[len(x)-2]
// 	tmp.Assign(value, h, w)
// }

// func (t Tensor) Shape() []int {
// 	shape := []int{}
// 	var tmp Tensor
// 	tmp = t
// 	for {
// 		if value, ok := tmp.(*Matrix); ok {
// 			h, w := value.Shape()
// 			shape = append(shape, []int{h, w}...)
// 		} else {
// 			shape = append(shape, len(tmp))
// 			tmp = tmp[0].(Tensor)
// 		}
// 	}
// 	return shape
// }

// func (t Tensor3D) Window(x, y, h, w int) Tensor3D {
// 	newT3D := Tensor3D{}
// 	for _, mat := range t {
// 		newT3D = append(newT3D, mat.Window(x, y, h, w))
// 	}
// 	return newT3D
// }

// func (t Tensor3D) Pad(size int) Tensor3D {
// 	newT3d := Tensor3D{}
// 	for _, m := range t {
// 		newT3d = append(newT3d, m.Pad(size))
// 	}
// 	return newT3d
// }

// func ZerosTensor(x ...int) Tensor {
// 	h := x[-1]
// 	w := x[-2]
// 	if len(x) == 2 {
// 		return Zeros(h, w)
// 	}
//
// 	zeros := []Tensor{}
// 	for i := 0; i < x[0]; i++ {
// 		ten = ZerosTensor(x[1:])
// 		zeros = append(zeros, ten)
// 	}
// 	return zeros
// }
