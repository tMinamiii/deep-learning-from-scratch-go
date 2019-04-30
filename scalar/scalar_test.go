package scalar

import (
	"log"
	"testing"
)

func TestRelu(t *testing.T) {
	if Relu(10) != 10 {
		t.Fail()
	}
	if Relu(1) != 1 {
		t.Fail()
	}
	if Relu(0) != 0 {
		t.Fail()
	}
	if Relu(-1) != 0 {
		t.Fail()
	}
}

func TestSigmoid(t *testing.T) {
	if Sigmoid(10) != 0.9999546021312976 {
		log.Println(Sigmoid(10))
		t.Fail()
	}
	if Sigmoid(1) != 0.7310585786300049 {
		log.Println(Sigmoid(1))
		t.Fail()
	}
	if Sigmoid(0) != 0.5 {
		log.Println(Sigmoid(0))
		t.Fail()
	}
	if Sigmoid(-1) != 0.2689414213699951 {
		log.Println(Sigmoid(-1))
		t.Fail()
	}
}

func TestStepFunction(t *testing.T) {
	if StepFunction(10) != 1 {
		t.Fail()
	}
	if StepFunction(1) != 1 {
		t.Fail()
	}
	if StepFunction(0) != 0 {
		t.Fail()
	}
	if StepFunction(-1) != 0 {
		t.Fail()
	}
}

func TestNumericalDiff1(t *testing.T) {
	diff := NumericalDiff(function1, 5)
	if diff != 0.1999999999990898 {
		log.Println(diff)
		t.Fail()
	}
}

func TestNumericalDiff2(t *testing.T) {
	diff := NumericalDiff(function1, 10)
	if diff != 0.2999999999997449 {
		log.Println(diff)
		t.Fail()
	}
}

func TestGradientDescent(t *testing.T) {
	//initX := []float64{-3.0 , 4.0}
	//GradientDescent()
}
