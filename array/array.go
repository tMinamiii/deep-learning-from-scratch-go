package array

import (
	"errors"
	"math"

	"github.com/naronA/zero_deeplearning/num"
)

func Equal(x1 []float64, x2 []float64) bool {
	if len(x1) != len(x2) {
		return false
	}
	for i := 0; i < len(x1); i++ {
		diff := x1[i] - x2[i]
		if diff < 0.0 && diff < -0.00001 {
			return false
		}
		if diff > 0.0 && diff > 0.00001 {
			return false
		}
	}
	return true
}

func Pow(x []float64, y float64) ([]float64, error) {
	result := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		result[i] = math.Pow(x[i], y)
	}
	return result, nil
}

func Multi(x1 []float64, x2 []float64) ([]float64, error) {
	if len(x1) != len(x2) {
		err := errors.New("not matched array length")
		return nil, err
	}
	result := make([]float64, len(x1))
	for i := 0; i < len(x1); i++ {
		result[i] = x1[i] * x2[i]
	}
	return result, nil
}

func Sum(ary []float64) float64 {
	sum := 0.0
	for _, num := range ary {
		sum += num
	}
	return sum
}

func Relu(x []float64) []float64 {
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = num.Relu(v)
	}
	return result
}

func Sigmoid(x []float64) []float64 {
	result := make([]float64, len(x))
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

func Max(x []float64) float64 {
	max := math.SmallestNonzeroFloat64
	for _, v := range x {
		max = math.Max(max, v)
	}
	return max
}

func Exp(x []float64) []float64 {
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Exp(v)
	}
	return result
}

func Sub(x []float64, y float64) []float64 {
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = v - y
	}
	return result
}

func Divide(x []float64, y float64) []float64 {
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = v / y
	}
	return result
}

func Softmax(a []float64) []float64 {
	c := Max(a)
	expA := Exp(Sub(a, c))
	sumExpA := Sum(expA)
	result := Divide(expA, sumExpA)
	return result
}

func IdentityFunction(x []float64) []float64 {
	return x
}

func MeanSquaredError(y, t float64) {
	return 0.5 * Sum((y - t))

}
