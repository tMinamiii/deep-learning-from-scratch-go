package num

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
	if Sigmoid(10) != 4.5397868702274976e-05 {
		log.Println(Sigmoid(10))
		t.Fail()
	}
	if Sigmoid(1) != 0.26894142136992605 {
		log.Println(Sigmoid(1))
		t.Fail()
	}
	if Sigmoid(0) != 0.5 {
		log.Println(Sigmoid(0))
		t.Fail()
	}
	if Sigmoid(-1) != 0.731058578630074 {
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
