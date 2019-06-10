package tensor

func (t *Tensor) Reshape(row, col int) *Tensor {
	if len(t.Shape) == 2 {
		m := t.Mat
		return &Tensor{
			Mat:   reshapeMat(m, row, col),
			Shape: []int{row, col},
		}
	}
	panic(t)
}

func reshapeMat(m *Matrix, row, col int) *Matrix {
	r, c := m.Shape()
	size := r * c
	if row == -1 {
		row = size / col
	} else if col == -1 {
		col = size / row
	}
	if m.Rows*m.Columns != row*col {
		return nil
	}

	return &Matrix{
		Vector:  m.Vector,
		Rows:    row,
		Columns: col,
	}
}

func (t *Tensor) Reshape2DTo4D(a, b, c, d int) *Tensor {
	if len(t.Shape) == 2 {
		m := t.Mat
		row, col := m.Shape()
		size := row * col
		switch {
		case a == -1:
			a = size / b / c / d
		case b == -1:
			b = size / a / c / d
		case c == -1:
			c = size / a / b / d
		case d == -1:
			d = size / a / b / c
		}

		t4d := ZerosT4D(a, b, c, d)
		for i := 0; i < a; i++ {
			for j := 0; j < b; j++ {
				sv := m.Vector[(i*b+j)*c*d : (i*b+j+1)*c*d]
				t4d[i][j] = &Matrix{
					Vector:  sv,
					Rows:    c,
					Columns: d,
				}
			}
		}

		return &Tensor{
			T4D:   t4d,
			Shape: []int{a, b, c, d},
		}
	}
	panic(t)
}

func (t *Tensor) Reshape2DTo5D(a, b, c, d, e int) *Tensor {
	if len(t.Shape) == 2 {
		m := t.Mat
		row, col := m.Shape()
		size := row * col
		switch {
		case a == -1:
			a = size / b / c / d / e
		case b == -1:
			b = size / a / c / d / e
		case c == -1:
			c = size / a / b / d / e
		case d == -1:
			d = size / a / b / c / e
		case e == -1:
			e = size / a / b / c / d
		}
		t5d := ZerosT5D(a, b, c, d, e)
		for i := 0; i < a; i++ {
			for j := 0; j < b; j++ {
				for k := 0; k < c; k++ {
					sv := m.Vector[((i*b+j)*c+k)*d*e : ((i*b+j)*c+k+1)*d*e]
					t5d[i][j][k] = &Matrix{
						Vector:  sv,
						Rows:    c,
						Columns: d,
					}
				}
			}
		}

		return &Tensor{
			T5D:   t5d,
			Shape: []int{a, b, c, d, e},
		}
	}
	panic(t)
}

func (t *Tensor) Reshape2DTo6D(a, b, c, d, e, f int) *Tensor {
	if len(t.Shape) == 2 {
		m := t.Mat
		t6d := ZerosT6D(a, b, c, d, e, f)
		for i := 0; i < a; i++ {
			for j := 0; j < b; j++ {
				for k := 0; k < c; k++ {
					for l := 0; l < d; l++ {
						sv := m.Vector[(((i*b+j)*c+k)*d+l)*e*f : (((i*b+j)*c+k)*d+l+1)*e*f]
						t6d[i][j][k][l] = &Matrix{
							Vector:  sv,
							Rows:    e,
							Columns: f,
						}
					}
				}
			}
		}
		return &Tensor{
			T6D:   t6d,
			Shape: []int{a, b, c, d, e, f},
		}
	}
	panic(t)
}

func (t *Tensor) Reshape4DTo2D(row, col int) *Tensor {
	if len(t.Shape) == 4 {
		t4d := t.T4D
		size := t4d.Size()
		if col == -1 {
			col = size / row
		} else if row == -1 {
			row = size / col
		}
		flat := t4d.Flatten()
		return &Tensor{
			Mat: &Matrix{
				Vector:  flat,
				Rows:    row,
				Columns: col,
			},
			Shape: t.Shape,
		}
	}
	panic(t)
}

func (t *Tensor) Reshape5DTo2D(row, col int) *Tensor {
	if len(t.Shape) == 5 {
		t5d := t.T5D
		a, b, c, d, e := t5d.Shape()
		size := a * b * c * d * e
		if row == -1 {
			row = size / col
		} else if col == -1 {
			col = size / row
		}

		return &Tensor{
			Mat: &Matrix{
				Vector:  t5d.Flatten(),
				Rows:    row,
				Columns: col,
			},
			Shape: []int{row, col},
		}
	}
	panic(t)
}
