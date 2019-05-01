package network

import (
	"math/rand"
	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
)

type TwoLayerNet struct {
	Params map[string]*mat.Mat64
}

func NewTwoLayerNet() {
	params := map[string]*mat.Mat64{}
	params["W1"] = weightInitStd * kkkk
}

