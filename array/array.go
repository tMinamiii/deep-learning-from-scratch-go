package array

import (
	"errors"
	"log"

	"github.com/naronA/zero_deeplearning/num"
)

func Equal(x1 []float64, x2 []float64) bool {
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
