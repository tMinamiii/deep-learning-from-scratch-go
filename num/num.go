package num

import (
	"math"
)

const E float64 = 2.71828182846

func Relu(x float64) float64 {
	if x <= 0.0 {
		return 0
	}
	return x
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func StepFunction(x float64) int {
	if x > 0 {
		return 1
	}
	return 0
}

func NumericalDiff(f func(float64) float64, x float64) float64 {
	h := 1e-4
	return (f(x+h) - f(x-h)) / (2 * h)
}
