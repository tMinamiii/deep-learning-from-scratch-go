package array

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/naronA/zero_deeplearning/scalar"
)

type Array []float64

const delta = 1e-7

func (x1 Array) Equal(x2 Array) bool {
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

func (x1 Array) NotEqual(x2 Array) bool {
	return !x1.Equal(x2)
}

func Randn(n int) Array {
	zeros := make(Array, n)
	for i := range zeros {
		zeros[i] = rand.NormFloat64()
	}
	return zeros
}

func Zeros(n int) Array {
	zeros := make(Array, n)
	for i := range zeros {
		zeros[i] = 0
	}
	return zeros
}

func ZerosLike(x Array) Array {
	zeros := make(Array, len(x))
	for i := range zeros {
		zeros[i] = 0
	}
	return zeros
}

func (x1 Array) Pow(x2 float64) Array {
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = math.Pow(x1[i], x2)
	}
	return result
}

func (x1 Array) Add(x2 Array) Array {
	if len(x1) != len(x2) {
		return nil
	}
	result := make(Array, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] + x2[i]
	}
	return result
}

func (x1 Array) Sub(x2 Array) Array {
	if len(x1) != len(x2) {
		return nil
	}
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] - x2[i]
	}
	return result
}

func (x1 Array) Multi(x2 Array) Array {
	if len(x1) != len(x2) {
		return nil
	}
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] * x2[i]
	}
	return result
}

func (x1 Array) Divide(x2 Array) Array {
	if len(x1) != len(x2) {
		return nil
	}
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] / x2[i]
	}
	return result
}

func (x1 Array) AddAll(x2 float64) Array {
	result := make(Array, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] + x2
	}
	return result
}

func (x1 Array) SubAll(x2 float64) Array {
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] - x2
	}
	return result
}

func (x1 Array) MultiAll(x2 float64) Array {
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] * x2
	}
	return result
}

func (x Array) DivideAll(y float64) Array {
	result := make(Array, len(x))
	for i, v := range x {
		result[i] = v / y
	}
	return result
}

func Sum(ary Array) float64 {
	sum := 0.0
	for _, num := range ary {
		sum += num
	}
	return sum
}

func Relu(x Array) Array {
	result := make(Array, len(x))
	for i, v := range x {
		result[i] = scalar.Relu(v)
	}
	return result
}

func Sigmoid(x Array) Array {
	result := make(Array, len(x))
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

func Max(x Array) float64 {
	max := math.SmallestNonzeroFloat64
	for _, v := range x {
		max = math.Max(max, v)
	}
	return max
}

func Exp(x Array) Array {
	result := make(Array, len(x))
	for i, v := range x {
		result[i] = math.Exp(v)
	}
	return result
}

func Log(x Array) Array {
	result := make(Array, len(x))
	for i, v := range x {
		result[i] = math.Log(v)
	}
	return result
}

func Softmax(x Array) Array {
	c := Max(x)
	expA := Exp(x.SubAll(c))
	sumExpA := Sum(expA)
	result := expA.DivideAll(sumExpA)
	return result
}

func IdentityFunction(x Array) Array {
	return x
}

func MeanSquaredError(y, t Array) float64 {
	sub := y.Sub(t)
	pow := sub.Pow(2)
	return 0.5 * Sum(pow)
}

func CrossEntropyError(y, t Array) float64 {
	log := Log(y.AddAll(delta))
	return -Sum(log.Multi(t))
}

func NumericalGradient(f func(Array) float64, x Array) Array {
	h := 1e-4
	grad := ZerosLike(x)

	fmt.Println(x)
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

func GradientDescent(f func(Array) float64, initX Array, lr float64, stepNum int) Array {
	x := initX
	for i := 0; i < stepNum; i++ {
		grad := NumericalGradient(f, x)
		x = grad.MultiAll(lr).Sub(x)
	}
	return x
}

func function1(x Array) float64 {
	return (0.01 * x[0] * x[0]) + 0.1*x[0]
}

func function2(x Array) float64 {
	return x[0]*x[0] + x[1]*x[1]
}
