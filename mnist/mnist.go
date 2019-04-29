package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"image"
	"image/color"
	"io"
	"os"
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width      = 28
	Height     = 28
)

type MnistDataSet struct {
	Images *MnistImage
	Label  []MnistLabel
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

	trainImages := readImages(trainImagesFile)
	trainLabels := readLabels(trainLabelsFile)
	testImages := readImages(testImagesFile)
	testLabels := readLabels(testLabelsFile)
	return &MnistDataSet{trainImages, trainLabels}, &MnistDataSet{testImages, testLabels}

}

type MnistLabel struct {
	number uint8
	oneHot []uint8
}

func oneHot(n uint8) []uint8 {
	oneHot := []uint8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	oneHot[n] = 1
	return oneHot
}

func readLabels(file *os.File) []MnistLabel {
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
	labels := make([]MnistLabel, n)
	for i := 0; i < int(n); i++ {
		var num uint8
		if err := binary.Read(r, binary.BigEndian, &num); err != nil {
			return nil
		}
		labels[i].number = num
		labels[i].oneHot = oneHot(num)
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

type MnistImage struct {
	rows int
	cols int
	imgs []RawImage
}

func readImages(file *os.File) *MnistImage {
	r, err := gzip.NewReader(file)
	if err != nil {
		return nil
	}
	defer r.Close()
	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)

	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return &MnistImage{0, 0, nil}
	}
	if magic != imageMagic {
		return &MnistImage{0, 0, nil}
	}
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return &MnistImage{0, 0, nil}
	}
	if err := binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return &MnistImage{0, 0, nil}
	}
	if err := binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return &MnistImage{0, 0, nil}
	}
	// N個のラベルデータが含まれているのでN要素の配列をつくる
	imgs := make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[1] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return &MnistImage{0, 0, nil}
		}
		if m_ != int(m) {
			return &MnistImage{0, 0, nil}
		}
	}
	return &MnistImage{int(nrow), int(ncol), imgs}
}
