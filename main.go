package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/mnist"
)

func train() {

	train, _ := mnist.LoadMnist()

	// ハイパーパラメタ
	const (
		ImageLength  = 784
		ItersNum     = 10000
		BatchSize    = 100
		LearningRate = 0.1
	)
	TrainSize := len(train.Labels)

	// net := network.NewTwoLayerNet(ImageLength, 50, 10, 0.01)

	rand.Seed(time.Now().UnixNano())
	for i := 0; i < ItersNum; i++ {
		batchIndices := rand.Perm(TrainSize)[:BatchSize]
		xBatch := make([]*mat.Matrix, TrainSize)
		for i, v := range batchIndices {
			image := train.Images[v]
			xBatch[i], err := mat.NewMat64(BatchSize, ImageLength, []float64(image))

			label := train.Labels[v]
		}
	}
}

func main() {

}
