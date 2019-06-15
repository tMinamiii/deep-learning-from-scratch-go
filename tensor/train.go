package tensor

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/naronA/zero_deeplearning/mnist"
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

func MnistTensor4D(set *mnist.DataSet) (*Tensor, *Tensor) {
	size := len(set.Labels)
	image := make(Tensor4D, size)
	label := make(vec.Vector, 0, size*len(set.Labels[0]))
	for i := 0; i < size; i++ {
		mat := &Matrix{
			Vector:  set.Images[i],
			Rows:    28,
			Columns: 28,
		}
		t3d := Tensor3D{mat}
		image[i] = t3d
		label = append(label, set.Labels[i]...)
	}
	a, b, c, d := image.Shape()
	t := &Tensor{
		Mat:   &Matrix{Rows: size, Columns: 10, Vector: label},
		Shape: []int{size, 10},
	}
	imageTen := &Tensor{
		T4D:   image,
		Shape: []int{a, b, c, d},
	}
	return imageTen, t
}

func MnistMatrix(set *mnist.DataSet) (*Tensor, *Tensor) {
	size := len(set.Labels)
	image := make(vec.Vector, 0, size*len(set.Images[0]))
	label := make(vec.Vector, 0, size*len(set.Labels[0]))
	for i := 0; i < size; i++ {
		image = append(image, set.Images[i]...)
		label = append(label, set.Labels[i]...)
	}
	x := &Tensor{
		Mat: &Matrix{
			Rows:    size,
			Columns: ImageLength,
			Vector:  image,
		},
		Shape: []int{size, ImageLength},
	}
	t := &Tensor{
		Mat: &Matrix{
			Rows:    size,
			Columns: 10,
			Vector:  label,
		},
		Shape: []int{size, 10},
	}
	return x, t
}

func train() {
	train, test := mnist.LoadMnist()
	TrainSize := len(train.Labels)
	opt := NewAdam(LearningRate)

	inputDim := &InputDim{
		Channel: 1,
		Height:  28,
		Weidth:  28,
	}

	convParams := &ConvParams{
		FilterNum:  30,
		FilterSize: 5,
		Pad:        0,
		Stride:     1,
	}

	net := NewSimpleConvNet(opt, inputDim, convParams, 100, 10, 0.01)

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
		xBatch := make(Tensor4D, BatchSize)
		for j, v := range batchIndices {
			mat := &Matrix{
				Vector:  train.Images[v],
				Rows:    28,
				Columns: 28,
			}
			t3d := Tensor3D{mat}
			xBatch[j] = t3d
			label = append(label, train.Labels[v]...)
		}
		tBatch := &Matrix{Rows: BatchSize, Columns: 10, Vector: label}
		e, f := tBatch.Shape()
		tBatchTen := &Tensor{Mat: tBatch, Shape: []int{e, f}}
		a, b, c, d := xBatch.Shape()
		xBatchTen := &Tensor{T4D: xBatch, Shape: []int{a, b, c, d}}
		grads := net.Gradient(xBatchTen, tBatchTen)
		net.UpdateParams(grads)
		loss := net.Loss(xBatchTen, tBatchTen)

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
