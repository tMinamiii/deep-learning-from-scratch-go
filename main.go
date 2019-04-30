package main

import (
	"image/color"
	"math/rand"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Points Example"
	p.X.Label.Text = "X value"
	p.Y.Label.Text = "Y value"

	p.Add(plotter.NewGrid())
	//クラス1
	x1, y1 := 8.0, 2.0

	//クラス2
	x2, y2 := 3.0, 6.0

	//各クラスのサンプル
	n := 200

	// 散布図の作成
	plot1, err := plotter.NewScatter(randomPoints(n, x1, y1))
	if err != nil {
		panic(err)
	}

	plot2, err := plotter.NewScatter(randomPoints(n, x2, y2))
	if err != nil {
		panic(err)
	}

	//色を指定する．
	plot1.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 55}
	plot2.GlyphStyle.Color = color.RGBA{R: 155, B: 128, A: 255}

	//plot1,plot2をplot
	p.Add(plot1)
	p.Add(plot2)

	//label
	p.Legend.Add("class1", plot1)
	p.Legend.Add("class2", plot2)

	// 座標範囲
	p.X.Min = 0
	p.X.Max = 10
	p.Y.Min = 0
	p.Y.Max = 10
	if err := p.Save(6*vg.Inch, 6*vg.Inch, "plot.png"); err != nil {
		panic(err)
	}
}

//ガウス分布
func random(axis float64) float64 {
	//分散
	dispersion := 1.0
	rand.Seed(time.Now().UnixNano())
	return rand.NormFloat64()*dispersion + axis
}

//学習データの生成
func randomPoints(n int, x, y float64) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := range pts {
		pts[i].X = random(x)
		pts[i].Y = random(y)
	}
	return pts
}
