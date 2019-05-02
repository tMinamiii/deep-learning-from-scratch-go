package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/naronA/zero_deeplearning/array"
	"github.com/naronA/zero_deeplearning/mat"
	"github.com/naronA/zero_deeplearning/mnist"
	"github.com/naronA/zero_deeplearning/network"
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

	net := network.NewTwoLayerNet(ImageLength, 50, 10, 0.01)

	trainLossList := []float64{}
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < ItersNum; i++ {
		batchIndices := rand.Perm(TrainSize)[:BatchSize]
		image := array.Array{}
		label := array.Array{}
		for _, v := range batchIndices {
			image = append(image, train.Images[v]...)
			label = append(label, train.Labels[v]...)
		}

		xBatch, _ := mat.NewMat64(BatchSize, ImageLength, image)
		tBatch, _ := mat.NewMat64(BatchSize, 10, label)
		grad := net.NumericalGradient(xBatch, tBatch)
		keys := []string{"W1", "b1", "W2", "b2"}
		for _, k := range keys {
			net.Params[k] = net.Params[k].Sub(grad[k].MulAll(LearningRate))
		}
		loss := net.Loss(xBatch, tBatch)
		fmt.Println(loss)
		trainLossList = append(trainLossList, loss)
	}
	fmt.Println(trainLossList)
}

func main() {
	train()
}
