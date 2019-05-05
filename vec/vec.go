package vec

import (
	"math"
	"math/rand"
	"time"

	"github.com/naronA/zero_deeplearning/scalar"
)

type Vector []float64

const delta = 1e-7

func Equal(x1, x2 Vector) bool {
	if len(x1) != len(x2) {
		return false
	}

	for i := 0; i < len(x2); i++ {
		diff := x1[i] - x2[i]
		if diff < 0.0 && diff < -delta {
			return false
		}
		if diff > 0.0 && diff > delta {
			return false
		}
	}
	return true
}

func NotEqual(x1, x2 Vector) bool {
	return !Equal(x1, x2)
}

func Randn(n int) Vector {
	rand.Seed(time.Now().UnixNano())
	zeros := make(Vector, n)
	for i := range zeros {
		zeros[i] = rand.NormFloat64()
	}
	return zeros
}

func Zeros(n int) Vector {
	zeros := make(Vector, n)
	return zeros
}

func ZerosLike(x Vector) Vector {
	zeros := make(Vector, len(x))
	return zeros
}

func (self Vector) Add(arg interface{}) Vector {
	switch x2 := arg.(type) {
	case Vector:
		if len(self) != len(x2) {
			return nil
		}
		result := make([]float64, len(self))
		for i := 0; i < len(self); i++ {
			result[i] = self[i] + x2[i]
		}
		return result
	case float64:
		result := make([]float64, len(self))
		for i := 0; i < len(self); i++ {
			result[i] = self[i] + x2
		}
		return result
	}
	return nil
}

func (self Vector) Sub(arg interface{}) Vector {
	switch x2 := arg.(type) {
	case Vector:
		if len(self) != len(x2) {
			return nil
		}
		result := make([]float64, len(self))
		for i := 0; i < len(self); i++ {
			result[i] = self[i] - x2[i]
		}
		return result
	case float64:
		result := make([]float64, len(self))
		for i := 0; i < len(self); i++ {
			result[i] = self[i] - x2
		}
		return result
	}
	return nil
}

func (self Vector) Mul(arg interface{}) Vector {
	switch x2 := arg.(type) {
	case Vector:
		if len(self) != len(x2) {
			return nil
		}
		result := make([]float64, len(self))
		for i := 0; i < len(self); i++ {
			result[i] = self[i] * x2[i]
		}
		return result
	case float64:
		result := make([]float64, len(self))
		for i := 0; i < len(self); i++ {
			result[i] = self[i] * x2
		}
		return result
	}
	return nil
}

func (self Vector) Div(arg interface{}) Vector {
	switch x2 := arg.(type) {
	case Vector:
		if len(self) != len(x2) {
			return nil
		}
		result := make([]float64, len(self))
		for i := 0; i < len(self); i++ {
			result[i] = self[i] / x2[i]
		}
		return result
	case float64:
		result := make([]float64, len(self))
		for i := 0; i < len(self); i++ {
			result[i] = self[i] / x2
		}
		return result
	}
	return nil
}

func Sum(ary Vector) float64 {
	sum := 0.0
	for _, num := range ary {
		sum += num
	}
	return sum
}

func Relu(x Vector) Vector {
	result := make(Vector, len(x))
	for i, v := range x {
		result[i] = scalar.Relu(v)
	}
	return result
}

func Sigmoid(x Vector) Vector {
	result := make(Vector, len(x))
	for i, v := range x {
		result[i] = scalar.Sigmoid(v)
	}
	return result
}

func StepFunction(x []float64) []int {
	result := make([]int, len(x))
	for i, v := range x {
		result[i] = scalar.StepFunction(v)
	}
	return result
}

func ArgMax(x Vector) int {
	max := math.SmallestNonzeroFloat64
	maxIndex := 0
	for i, v := range x {
		if max != math.Max(max, v) {
			maxIndex = i
			max = math.Max(max, v)
		}
	}
	return maxIndex
}

func Max(x Vector) float64 {
	max := math.SmallestNonzeroFloat64
	for _, v := range x {
		max = math.Max(max, v)
	}
	return max
}

func Exp(x Vector) Vector {
	result := make(Vector, len(x))
	for i, v := range x {
		result[i] = math.Exp(v)
	}
	return result
}

func Log(x Vector) Vector {
	result := make(Vector, len(x))
	for i, v := range x {
		result[i] = math.Log(v)
	}
	return result
}

func Pow(x Vector, p float64) Vector {
	result := make(Vector, len(x))
	for i, v := range x {
		result[i] = math.Pow(v, p)
	}
	return result
}

func Sqrt(x Vector) Vector {
	result := make(Vector, len(x))
	for i, v := range x {
		result[i] = math.Sqrt(v)
	}
	return result
}

func Abs(x Vector) Vector {
	result := make(Vector, len(x))
	for i, v := range x {
		result[i] = math.Abs(v)
	}
	return result
}

/*
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))
Vectorのsoftmaxなので常にndim == 1
*/
func Softmax(x Vector) Vector {
	c := Max(x)
	expA := Exp(x.Sub(c))
	sumExpA := Sum(expA)
	result := expA.Div(sumExpA)
	return result
}

func IdentityFunction(x Vector) Vector {
	return x
}

func MeanSquaredError(y, t Vector) float64 {
	sub := y.Sub(t)
	pow := Pow(sub, 2)
	return 0.5 * Sum(pow)
}

func CrossEntropyError(y, t Vector) float64 {
	log := Log(y.Add(delta))
	return -Sum(log.Mul(t))
}

func NumericalGradient(f func(Vector) float64, x Vector) Vector {
	h := 1e-4
	grad := ZerosLike(x)

	for idx := range x {
		tmpVal := x[idx]
		// f(x+h)の計算
		x[idx] = tmpVal + h
		fxh1 := f(x)

		// f(x+h)の計算
		x[idx] = tmpVal - h
		fxh2 := f(x)

		grad[idx] = (fxh1 - fxh2) / (2 * h)
		x[idx] = tmpVal
	}

	return grad
}

func GradientDescent(f func(Vector) float64, initX Vector, lr float64, stepNum int) Vector {
	x := initX
	for i := 0; i < stepNum; i++ {
		grad := NumericalGradient(f, x)
		x = grad.Mul(lr).Sub(x)
	}
	return x
}

func function1(x Vector) float64 {
	return (0.01 * x[0] * x[0]) + 0.1*x[0]
}

func function2(x Vector) float64 {
	return x[0]*x[0] + x[1]*x[1]
}
