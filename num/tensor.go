package num

type Tensor []interface{}

func (t Tensor) Flatten() vec.Vector {
	v := vec.Vector{}
	for _, e := range t {
		v = append(v, e.Vector...)
	}
	return v
}


func (t Tensor) Element(x ...int) float64 {
	tmp := t
	for i := 0 ; i < len(x) - 2; i++ {
		tmp = tmp[x[i]]
	}
	h := x[-1]
	w := x[-2]
	return tmp.Element(h, w)
}

func (t Tensor) Assign(value float64, x ...int) {
	tmp := t
	for i := 0 ; i < len(x) - 2; i++ {
		tmp = tmp[x[i]]
	}
	h := x[-1]
	w := x[-2]
	tmp.Assign(value, h, w)
}

func (t Tensor) Shape() []int {
	shape := []int
	tmp := t
	for ;; {
		if value, ok := tmp.(*Matrix); ok {
			h, w := value.Shape()
			shape = append(shape, []int{h,w}...)
		} else {
			shape = append(shape, len(tmp))
			tmp = tmp[0]
		}
	}
	return shape
}

// func (t Tensor3D) Window(x, y, h, w int) Tensor3D {
// 	newT3D := Tensor3D{}
// 	for _, mat := range t {
// 		newT3D = append(newT3D, mat.Window(x, y, h, w))
// 	}
// 	return newT3D
// }

func (t Tensor3D) Pad(size int) Tensor3D {
	newT3d := Tensor3D{}
	for _, m := range t {
		newT3d = append(newT3d, m.Pad(size))
	}
	return newT3d
}

func ZerosTensor(x ...int) Tensor {
	h := x[-1]
	w := x[-2]
	if (len(x) == 2) {
		return Zeros(h, w)
	}

	zeros := []Tensor{}
	for i := 0 ; i < x[0]; i++ {
		ten = ZerosTensor(x[1:])
		zeros = append(zeros, ten)
	}
	return zeros
}

func calcMats(a ArithmeticT3D, x1 interface{}, x2 interface{}) *Matrix {
	switch a {
	case ADDT3D:
		return Add(x1, x2)
	case SUBT3D:
		return Sub(x1, x2)
	case MULT3D:
		return Mul(x1, x2)
	case DIVT3D:
		return Div(x1, x2)
	}
	return nil
}

func t3dT3d(a ArithmeticT3D, x1 Tensor3D, x2 Tensor3D) Tensor3D {
	mats := make(Tensor3D, len(x1))
	x1mat := x1
	x2mat := x2
	for i := range x1 {
		mats[i] = calcMats(a, x1mat[i], x2mat[i])
	}
	return mats
}

func t3dAny(a ArithmeticT3D, x1 Tensor3D, x2 interface{}) Tensor3D {
	mats := make(Tensor3D, len(x1))
	for i, x1mat := range x1 {
		mats[i] = calcMats(a, x1mat, x2)
	}
	return mats
}

func anyT3d(a ArithmeticT3D, x1 interface{}, x2 Tensor3D) Tensor3D {
	tensor := ZerosLikeT3D(x2)
	mats := tensor
	for i, x2mat := range x2 {
		mats[i] = calcMats(a, x1, x2mat)
	}
	return tensor
}

type ArithmeticT3D int

const (
	ADDT3D ArithmeticT3D = iota
	SUBT3D
	MULT3D
	DIVT3D
)

func calcArithmetic(a ArithmeticT3D, x1 interface{}, x2 interface{}) Tensor3D {
	if x1v, ok := x1.(Tensor3D); ok {
		switch x2v := x2.(type) {
		case Tensor3D:
			return t3dT3d(a, x1v, x2v)
		case *Matrix:
		case vec.Vector:
		case float64:
			return t3dAny(a, x1v, x2v)
		case int:
			return t3dAny(a, x1v, float64(x2v))
		}
	} else if x2v, ok := x2.(Tensor3D); ok {
		switch x1v := x1.(type) {
		case *Matrix:
		case vec.Vector:
		case float64:
			return anyT3d(a, x1v, x2v)
		case int:
			return anyT3d(a, float64(x1v), x2v)
		}
	}
	return nil
}

func AddT3D(x1 interface{}, x2 interface{}) Tensor3D {
	return calcArithmetic(ADDT3D, x1, x2)
}

func SubT3D(x1 interface{}, x2 interface{}) Tensor3D {
	return calcArithmetic(SUBT3D, x1, x2)
}

func MulT3D(x1 interface{}, x2 interface{}) Tensor3D {
	return calcArithmetic(MULT3D, x1, x2)
}

func DivT3D(x1 interface{}, x2 interface{}) Tensor3D {
	return calcArithmetic(DIVT3D, x1, x2)
}

func EqualT3D(t1, t2 Tensor3D) bool {
	for i := range t1 {
		if NotEqual(t1[i], t2[i]) {
			return false
		}
	}
	return true
}
