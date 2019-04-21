package num

import (
	"errors"
	"log"
	"math"
)

const e float64 = 2.71828182846

func Relu(x float64) float64 {
	if x <= 0.0 {
		return 0
	}
	return x
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + Exp(x))
}

func Exp(x float64) float64 {
	return math.Pow(e, x)
}

// ArrayEqual is
func ArrayEqual(x1 []float64, x2 []float64) bool {
	if len(x1) != len(x2) {
		log.Println("length not matched")
		return false
	}
	for i := 0; i < len(x1); i++ {
		diff := x1[i] - x2[i]
		if diff < 0.0 && diff < -0.00001 {
			log.Printf("not equals %f < %f", x1[i], x2[i])
			return false
		}
		if diff > 0.0 && diff > 0.00001 {
			log.Printf("not equals %f > %f", x1[i], x2[i])
			return false
		}
	}
	return true
}

// ArrayMulti is
func ArrayMulti(x1 []float64, x2 []float64) ([]float64, error) {
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

// ArraySum is
func ArraySum(ary []float64) float64 {
	sum := 0.0
	for _, num := range ary {
		sum += num
	}
	return sum
}
