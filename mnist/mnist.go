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

func main() {
	// Read Label file
	file, err := os.Open("./mnsit/train-labels-idx1-ubyte.gz")
	if err != nil {
		return
	}
	defer file.Close()
	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return
	}
	defer gzipReader.Close()
}

type Label uint8

func readLabels(r io.Reader) []Label {
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
	labels := make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil
		}
		labels[i] = l
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
		Min: image.Point{0,0},
		Max: image.Point{Width, Height},
	}
}

type MnistImage struct {
	rows int
	cols int
	imgs []RawImage
}

func readImages(r io.Reader) *MnistImage {
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
