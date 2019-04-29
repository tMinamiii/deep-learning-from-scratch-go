package main

import (
	"fmt"

	"github.com/naronA/zero_deeplearning/mnist"
)

func main() {
	train, _ := mnist.LoadMnist()
	for i:=0 ; i<30; i++ {
		fmt.Println(train.Label[i])
	}
}
