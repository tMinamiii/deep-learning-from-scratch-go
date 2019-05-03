package network

import (
	"fmt"
	"testing"
)

func TestLayer(t *testing.T) {
	apple := 100.0
	appleNum := 2.0
	orange := 150.0
	orangeNum := 3.0
	tax := 1.1

	// layer
	mulAppleLayer := &MulLayer{}
	mulOrangeLayer := &MulLayer{}
	addAppleOrangeLayer := &AddLayer{}
	mulTaxLayer := &MulLayer{}

	// forward
	applePrice := mulAppleLayer.forward(apple, appleNum)
	orangePrice := mulOrangeLayer.forward(orange, orangeNum)
	allPrice := addAppleOrangeLayer.forward(applePrice, orangePrice)
	price := mulTaxLayer.forward(allPrice, tax)

	// backward
	dprice := 1.0
	dallPrice, dtax := mulTaxLayer.backward(dprice)
	dapplePrice, dorangePrice := addAppleOrangeLayer.backward(dallPrice)
	dorange, dorangeNum := mulOrangeLayer.backward(dorangePrice)
	dapple, dappleNum := mulAppleLayer.backward(dapplePrice)

	if price != 715.0000000000001 {
		fmt.Println(price)
		t.Fail()
	}

	if dapple != 2.2 && dappleNum != 110 && dorange != 165 && dorangeNum != 3.3 && dtax != 650 {
		fmt.Println(dapple, dappleNum, dtax)
		t.Fail()
	}
}
