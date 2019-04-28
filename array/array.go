package array

import (
	"math"

	"github.com/naronA/zero_deeplearning/num"
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

func (x1 Array) Pow(x2 float64) Array {
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = math.Pow(x1[i], x2)
	}
	return result
}

func (x1 Array) Multi(x2 float64) Array {
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] * x2
	}
	return result
}

func (x1 Array) MultiArray(x2 Array) Array {
	if len(x1) != len(x2) {
		return nil
	}
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] * x2[i]
	}
	return result
}

func (x Array) Divide(y float64) Array {
	result := make(Array, len(x))
	for i, v := range x {
		result[i] = v / y
	}
	return result
}

func (x1 Array) DivideArray(x2 Array) Array {
	if len(x1) != len(x2) {
		return nil
	}
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] / x2[i]
	}
	return result
}

func (x1 Array) Add(x2 float64) Array {
	result := make(Array, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] + x2
	}
	return result
}

func (x1 Array) AddArray(x2 Array) Array {
	if len(x1) != len(x2) {
		return nil
	}
	result := make(Array, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] + x2[i]
	}
	return result
}

func (x1 Array) Sub(x2 float64) Array {
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] - x2
	}
	return result
}

func (x1 Array) SubArray(x2 Array) Array {
	if len(x1) != len(x2) {
		return nil
	}
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] - x2[i]
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
		result[i] = num.Relu(v)
	}
	return result
}

func Sigmoid(x Array) Array {
	result := make(Array, len(x))
	for i, v := range x {
		result[i] = num.Sigmoid(v)
	}
	return result
}

func StepFunction(x []float64) []int {
	result := make([]int, len(x))
	for i, v := range x {
		result[i] = num.StepFunction(v)
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

func Softmax(a Array) Array {
	c := Max(a)
	expA := Exp(a.Sub(c))
	sumExpA := Sum(expA)
	result := expA.Divide(sumExpA)
	return result
}

func IdentityFunction(x Array) Array {
	return x
}

func MeanSquaredError(y, t Array) float64 {
	sub := y.SubArray(t)
	pow := sub.Pow(2)
	return 0.5 * Sum(pow)
}

func CrossEntropyError(y, t Array) float64 {
	log := Log(y.Add(delta))
	return -Sum(log.MultiArray(t))
}
