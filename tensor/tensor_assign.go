package tensor

import "github.com/naronA/zero_deeplearning/tensor/types"

/* Depenedent*/
func (t *Tensor) Assign(value float64, point []int) {
	switch {
	case len(t.Shape) == 2:
		assignMat(t.Mat, value, point)
	case len(t.Shape) == 3:
		assignT3D(t.T3D, value, point)
	case len(t.Shape) == 4:
		assignT4D(t.T4D, value, point)
	case len(t.Shape) == 5:
		assignT5D(t.T5D, value, point)
	case len(t.Shape) == 6:
		assignT6D(t.T6D, value, point)
	}
	panic(t)
}

func assignMat(m *types.Matrix, value float64, point []int) {
	a := point[0]
	b := point[1]
	m.Vector[a*m.Columns+b] = value
}

func assignT3D(t types.Tensor3D, value float64, point []int) {
	a := point[0]
	assignMat(t[a], value, point[1:])

}
func assignT4D(t types.Tensor4D, value float64, point []int) {
	a := point[0]
	assignT3D(t[a], value, point[1:])
}

func assignT5D(t types.Tensor5D, value float64, point []int) {
	a := point[0]
	assignT4D(t[a], value, point[1:])
}

func assignT6D(t types.Tensor6D, value float64, point []int) {
	a := point[0]
	assignT5D(t[a], value, point[1:])
}

func (t *Tensor) AssignWindow(window *types.Matrix, x, y, h, w int) {
	if len(t.Shape) == 2 {
		m := t.Mat
		assignMatWindow(m, window, x, y, h, w)
	}
	panic(t)
}

func assignMatWindow(m *types.Matrix, window *types.Matrix, x, y, h, w int) {
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			val := window.Element(i, j)
			assignMat(m, val, []int{i + x, j + y})
		}
	}
}

func AddAssign(t1 *types.Tensor4DSlice, t2 *Tensor) {
	if len(t2.Shape) == 4 {
		t4d := t2.T4D
		t2flat := t4d.Flatten()
		for i, idx := range t1.Indices {
			add := t1.Actual[idx.N][idx.C].Element(idx.H, idx.W) + t2flat[i]
			t1.Actual[idx.N][idx.C].Assign(add, idx.H, idx.W)
		}
	}
	panic(t2)
}
