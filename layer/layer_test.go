package layer

import (
	"fmt"
	"testing"

	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/vec"
)

func TestRelu(t *testing.T) {
	m := num.NewMatrix(vec.Vector{
		0, 1, -1,
		1, -1, 2,
	}, 2, 3)
	relu := NewRelu()
	actual := relu.Forward(m, true)
	expected := num.NewMatrix(vec.Vector{
		0, 1, 0,
		1, 0, 2,
	}, 2, 3)

	if num.NotEqual(actual, expected) {
		t.Fail()
	}

	actual = relu.Backward(m)

	if num.NotEqual(actual, expected) {
		t.Fail()
	}
}

func TestAffine(t *testing.T) {
	w := num.NewMatrix(vec.Vector{
		1, 0,
		0, 1,
	}, 2, 2)
	b := num.NewMatrix(vec.Vector{
		1, 1,
	}, 1, 2)
	affine := NewAffine(w, b)

	m := num.NewMatrix(vec.Vector{
		1, 2,
		3, 4,
	}, 2, 2)
	actual := affine.Forward(m, true)
	expected := num.NewMatrix(vec.Vector{
		2, 3,
		4, 5,
	}, 2, 2)

	if num.NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
	actual = affine.Backward(expected)

	if num.NotEqual(actual, expected) {
		fmt.Println(actual, expected)
		t.Fail()
	}
}
func TestSoftmaxWithLoss_1(t *testing.T) {
	softmax := NewSfotmaxWithLoss()
	xm := num.NewMatrix(vec.Vector{
		1, 0,
		0, 1,
	}, 2, 2)
	tm := num.NewMatrix(vec.Vector{
		1, 0,
		1, 0,
	}, 2, 2)

	actual := softmax.Forward(xm, tm)
	expected := 0.8132614332101986
	if actual != expected {
		fmt.Println(actual, expected)
		t.Fail()
	}

	expectedBackward := num.NewMatrix(vec.Vector{
		-0.13447071068499755, 0.13447071068499755,
		-0.36552928931500245, 0.36552928931500245,
	}, 2, 2)
	actualBackward := softmax.Backward(1.0)
	if num.NotEqual(actualBackward, expectedBackward) {
		fmt.Println(actualBackward, expectedBackward)
		t.Fail()
	}

}
func TestSoftmaxWithLoss_2(t *testing.T) {
	softmax := NewSfotmaxWithLoss()
	xm := num.NewMatrix(vec.Vector{
		1, 0,
		0, 1,
	}, 2, 2)
	tm := num.NewMatrix(vec.Vector{
		1, 0,
		0, 1,
	}, 2, 2)

	actual := softmax.Forward(xm, tm)
	expected := 0.3132615507302881
	if actual != expected {
		fmt.Println(actual, expected)
		t.Fail()
	}

	expectedBackward := num.NewMatrix(vec.Vector{
		-0.13447071068499755, 0.13447071068499755,
		0.13447071068499755, -0.13447071068499755,
		// 0.7310585786300049, 0.2689414213699951,
		// 0.2689414213699951, 0.7310585786300049,
	}, 2, 2)
	actualBackward := softmax.Backward(1.0)
	if num.NotEqual(actualBackward, expectedBackward) {
		fmt.Println(actualBackward, expectedBackward)
		t.Fail()
	}

}
