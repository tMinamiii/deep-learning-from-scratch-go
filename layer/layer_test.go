package layer

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/vec"
)

func TestRelu(t *testing.T) {
	m, _ := mat.NewMatrix(2, 3, vec.Vector{
		0, 1, -1,
		1, -1, 2,
	})
	relu := NewRelu()
	actual := relu.Forward(m)
	expected, _ := mat.NewMatrix(2, 3, vec.Vector{
		0, 1, 0,
		1, 0, 2,
	})

	if mat.NotEqual(actual, expected) {
		t.Fail()
	}

	actual = relu.Backward(m)

	if mat.NotEqual(actual, expected) {
		t.Fail()
	}
}

func TestAffine(t *testing.T) {
	w, _ := mat.NewMatrix(2, 2, vec.Vector{
		1, 0,
		0, 1,
	})
	b, _ := mat.NewMatrix(1, 2, vec.Vector{
		1, 1,
	})
	affine := NewAffine(w, b)

	m, _ := mat.NewMatrix(2, 2, vec.Vector{
		1, 2,
		3, 4,
	})
	actual := affine.Forward(m)
	expected, _ := mat.NewMatrix(2, 2, vec.Vector{
		2, 3,
		4, 5,
	})

	if mat.NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
	actual = affine.Backward(expected)

	if mat.NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}
func TestSoftmaxWithLoss_1(t *testing.T) {
	softmax := NewSfotmaxWithLoss()
	xm, _ := mat.NewMatrix(2, 2, vec.Vector{
		1, 0,
		0, 1,
	})
	tm, _ := mat.NewMatrix(2, 2, vec.Vector{
		1, 0,
		1, 0,
	})

	actual := softmax.Forward(xm, tm)
	expected := 0.8132614332101986
	if actual != expected {
		fmt.Println(actual, expected)
		t.Fail()
	}

	expectedBackward, _ := mat.NewMatrix(2, 2, vec.Vector{
		-0.13447071068499755, 0.13447071068499755,
		-0.36552928931500245, 0.36552928931500245,
	})
	actualBackward := softmax.Backward(1.0)
	if mat.NotEqual(actualBackward, expectedBackward) {
		fmt.Println(actualBackward, expectedBackward)
		t.Fail()
	}

}
func TestSoftmaxWithLoss_2(t *testing.T) {
	softmax := NewSfotmaxWithLoss()
	xm, _ := mat.NewMatrix(2, 2, vec.Vector{
		1, 0,
		0, 1,
	})
	tm, _ := mat.NewMatrix(2, 2, vec.Vector{
		1, 0,
		0, 1,
	})

	actual := softmax.Forward(xm, tm)
	expected := 0.3132615507302881
	if actual != expected {
		fmt.Println(actual, expected)
		t.Fail()
	}

	expectedBackward, _ := mat.NewMatrix(2, 2, vec.Vector{
		-0.13447071068499755, 0.13447071068499755,
		0.13447071068499755, -0.13447071068499755,
		// 0.7310585786300049, 0.2689414213699951,
		// 0.2689414213699951, 0.7310585786300049,
	})
	actualBackward := softmax.Backward(1.0)
	if mat.NotEqual(actualBackward, expectedBackward) {
		fmt.Println(actualBackward, expectedBackward)
		t.Fail()
	}

}
