package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/naronA/zero_deeplearning/mnist"
	"github.com/naronA/zero_deeplearning/network"
	"github.com/naronA/zero_deeplearning/num"
	"github.com/naronA/zero_deeplearning/optimizer"
	"github.com/naronA/zero_deeplearning/vec"
)

// ハイパーパラメタ
const (
	ImageLength  = 784
	ItersNum     = 100000
	BatchSize    = 100
	Hidden       = 100
	LearningRate = 0.0001
	MNIST        = 10
)

func MnistTensor4D(set *mnist.DataSet) (num.Tensor4D, *num.Matrix) {
	size := len(set.Labels)
	image := make(num.Tensor4D, size)
	// label := vec.Vector{}
	label := make(vec.Vector, 0, size*len(set.Labels[0]))
	for i := 0; i < size; i++ {
		mat := &num.Matrix{
			Vector:  set.Images[i],
			Rows:    28,
			Columns: 28,
		}
		t3d := num.Tensor3D{mat}
		image[i] = t3d
		label = append(label, set.Labels[i]...)
	}

	t, _ := num.NewMatrix(size, 10, label)
	return image, t
}

func MnistMatrix(set *mnist.DataSet) (*num.Matrix, *num.Matrix) {
	size := len(set.Labels)
	image := make(vec.Vector, 0, size*len(set.Images[0]))
	label := make(vec.Vector, 0, size*len(set.Labels[0]))
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
	opt := optimizer.NewAdamAny(LearningRate)
	// weightDecayLambda := 0.1

	inputDim := &network.InputDim{
		Channel: 1,
		Height:  28,
		Weidth:  28,
	}

	convParams := &network.ConvParams{
		FilterNum:  30,
		FilterSize: 5,
		Pad:        0,
		Stride:     1,
	}

	net := network.NewSimpleConvNet(opt, inputDim, convParams, 100, 10, 0.01)

	xTrain, tTrain := MnistTensor4D(train)
	xTest, tTest := MnistTensor4D(test)
	iterPerEpoch := func() int {
		return 50
		// if TrainSize/BatchSize > 1.0 {
		// 	return TrainSize / BatchSize
		// }
		// return 1
	}()

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < ItersNum; i++ {
		start := time.Now()
		batchIndices := rand.Perm(TrainSize)[:BatchSize]
		label := make(vec.Vector, 0, len(train.Labels[0])*BatchSize)
		xBatch := make(num.Tensor4D, BatchSize)
		for j, v := range batchIndices {
			mat := &num.Matrix{
				Vector:  train.Images[v],
				Rows:    28,
				Columns: 28,
			}
			t3d := num.Tensor3D{mat}
			xBatch[j] = t3d
			label = append(label, train.Labels[v]...)
		}
		tBatch, _ := num.NewMatrix(BatchSize, 10, label)

		grads := net.Gradient(xBatch, tBatch)
		net.UpdateParams(grads)
		loss := net.Loss(xBatch, tBatch)

		if i%iterPerEpoch == 0 && i >= iterPerEpoch {
			fmt.Println("calc accuracy")
			testAcc := net.Accuracy(xTest, tTest)
			trainAcc := net.Accuracy(xTrain, tTrain)
			end := time.Now()
			fmt.Printf("elapstime = %v loss = %v\n", end.Sub(start), loss)
			// fmt.Printf("test acc = %v \n", testAcc)
			fmt.Printf("train acc / test acc = %v / %v\n", trainAcc, testAcc)
		}
	}
}

func main() {
	train()
}
