package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"image"
	"image/color"
	"io"
	"os"

	"github.com/naronA/zero_deeplearning/array"
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width      = 28
	Height     = 28
)

type MnistDataSet struct {
	Rows      int
	Cols      int
	RawImages []RawImage
	Images    []array.Array
	Labels    []array.Array
}

func LoadMnist() (*MnistDataSet, *MnistDataSet) {

	trainImagesFile, err := os.Open("./mnist/train-images-idx3-ubyte.gz")
	if err != nil {
		return nil, nil
	}
	defer trainImagesFile.Close()
	trainLabelsFile, err := os.Open("./mnist/train-labels-idx1-ubyte.gz")
	if err != nil {
		return nil, nil
	}
	defer trainLabelsFile.Close()
	testImagesFile, err := os.Open("./mnist/t10k-images-idx3-ubyte.gz")
	if err != nil {
		return nil, nil
	}
	defer testImagesFile.Close()
	testLabelsFile, err := os.Open("./mnist/t10k-labels-idx1-ubyte.gz")
	if err != nil {
		return nil, nil
	}
	defer testLabelsFile.Close()

	trainRows, trainColumns, trainImages, trainFImages := readImages(trainImagesFile)
	trainLabels := readLabels(trainLabelsFile)
	train := &MnistDataSet{
		Rows:      trainRows,
		Cols:      trainColumns,
		RawImages: trainImages,
		Images:    trainFImages,
		Labels:    trainLabels,
	}
	testRows, testColumns, testImages, testFImages := readImages(testImagesFile)
	testLabels := readLabels(testLabelsFile)
	test := &MnistDataSet{
		Rows:      testRows,
		Cols:      testColumns,
		RawImages: testImages,
		Images:    testFImages,
		Labels:    testLabels,
	}
	return train, test
}

func oneHot(n uint8) array.Array {
	oneHot := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	oneHot[n] = 1
	return oneHot
}

func readLabels(file *os.File) []array.Array {
	r, err := gzip.NewReader(file)
	if err != nil {
		return nil
	}
	defer r.Close()

	var (
		magic int32
		n     int32
	)
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil
	}
	if magic != labelMagic {
		return nil
	}
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil
	}
	// N個のラベルデータが含まれているのでN要素の配列をつくる
	labels := make([]array.Array, n)
	for i := 0; i < int(n); i++ {
		var num uint8
		if err := binary.Read(r, binary.BigEndian, &num); err != nil {
			return nil
		}
		// labels[i].number = num
		// labels[i].oneHot = oneHot(num)
		labels[i] = oneHot(num)
	}
	return labels
}

type RawImage []byte

func (img RawImage) ColorModel() color.Model {
	return color.GrayModel
}

func (img RawImage) At(x, y int) color.Color {
	return color.Gray{img[y*Width+x]}
}

func (img RawImage) Bounds() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{0, 0},
		Max: image.Point{Width, Height},
	}
}

func readImages(file *os.File) (int, int, []RawImage, []array.Array) {
	r, err := gzip.NewReader(file)
	if err != nil {
		panic(err)
	}
	defer r.Close()
	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)

	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		panic(err)
	}
	if magic != imageMagic {
		panic(err)
	}
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		panic(err)
	}
	if err := binary.Read(r, binary.BigEndian, &nrow); err != nil {
		panic(err)
	}
	if err := binary.Read(r, binary.BigEndian, &ncol); err != nil {
		panic(err)
	}
	// N個のラベルデータが含まれているのでN要素の配列をつくる
	imgs := make([]RawImage, n)
	fimgs := make([]array.Array, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m)
		fimgs[i] = make(array.Array, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			panic(err)
		}
		if m_ != int(m) {
			return 0, 0, nil, nil
		}

		for j, b := range imgs[i] {
			fimgs[i][j] = float64(b)
		}
	}
	return int(nrow), int(ncol), imgs, fimgs
}
