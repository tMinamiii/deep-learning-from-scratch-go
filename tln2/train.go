package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/mnist"
	"github.com/naronA/zero_deeplearning/network"
	"github.com/naronA/zero_deeplearning/optimizer"
	"github.com/naronA/zero_deeplearning/vec"
)

// ハイパーパラメタ
const (
	ImageLength  = 784
	ItersNum     = 100000
	BatchSize    = 100
	Hidden       = 50
	LearningRate = 0.0001
	MNIST        = 10
)

func MnistMatrix(set *mnist.DataSet) (*num.Matrix, *num.Matrix) {
	size := len(set.Labels)
	image := vec.Vector{}
	label := vec.Vector{}
	for i := 0; i < size; i++ {
		image = append(image, set.Images[i]...)
		label = append(label, set.Labels[i]...)
	}
	x, _ := num.NewMatrix(size, ImageLength, image)
	t, _ := num.NewMatrix(size, 10, label)
	return x, t
}

func train() {

	train, test := mnist.LoadMnist()

	TrainSize := len(train.Labels)
	// opt := optimizer.NewSGD(LearningRate)
	opt := optimizer.NewMomentum(LearningRate)
	// opt := optimizer.NewAdaGrad(LearningRate)
	// opt := optimizer.NewAdam(LearningRate)
	weightDecayLambda := 0.1
	// net := network.NewMultiLayer(opt, ImageLength, Hidden, MNIST, weightDecayLambda)
	// net := network.NewTwoLayerNet(opt, ImageLength, Hidden, MNIST, weightDecayLambda)
	net := network.NewThreeLayerNet(opt, ImageLength, Hidden, MNIST, weightDecayLambda)

	xTrain, tTrain := MnistMatrix(train)
	xTest, tTest := MnistMatrix(test)
	iterPerEpoch := func() int {
		if TrainSize/BatchSize > 1.0 {
			return TrainSize / BatchSize
		}
		return 1
	}()

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < ItersNum; i++ {
		start := time.Now()
		batchIndices := rand.Perm(TrainSize)[:BatchSize]
		image := vec.Vector{}
		label := vec.Vector{}
		for _, v := range batchIndices {
			image = append(image, train.Images[v]...)
			label = append(label, train.Labels[v]...)
		}

		xBatch, _ := num.NewMatrix(BatchSize, ImageLength, image)
		tBatch, _ := num.NewMatrix(BatchSize, 10, label)
		grads := net.Gradient(xBatch, tBatch)
		net.UpdateParams(grads)
		loss := net.Loss(xBatch, tBatch, false)

		if i%iterPerEpoch == 0 && i >= iterPerEpoch {
			trainAcc := net.Accuracy(xTrain, tTrain)
			testAcc := net.Accuracy(xTest, tTest)
			end := time.Now()
			fmt.Printf("elapstime = %v loss = %v\n", end.Sub(start), loss)
			fmt.Printf("train acc / test acc = %v / %v\n", trainAcc, testAcc)
		}
	}
}

func main() {
	train()
}
