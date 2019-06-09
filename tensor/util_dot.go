package tensor

import (
	"github.com/naronA/zero_deeplearning/tensor/types"
	"github.com/naronA/zero_deeplearning/vec"
)

func dotMatPart(i int, a, b, c *types.Matrix, ch chan int) {
	ac := a.Columns
	bc := b.Columns
	for j := 0; j < bc; j++ {
		part := 0.0
		for k := 0; k < ac; k++ {
			part += a.Vector[i*a.Columns+k] * b.Vector[k*b.Columns+j]
		}
		c.Vector[i*c.Columns+j] = part
	}
	ch <- i
}

func dotMat(m1, m2 *types.Matrix) *types.Matrix {
	if m1.Columns != m2.Rows {
		return nil
	}
	v3 := vec.Zeros(m1.Rows * m2.Columns)
	m3 := &types.Matrix{
		Vector:  v3,
		Rows:    m1.Rows,
		Columns: m2.Columns,
	}

	ch := make(chan int)
	for i := 0; i < m1.Rows; i++ {
		go dotMatPart(i, m1, m2, m3, ch)
	}
	for i := 0; i < m1.Rows; i++ {
		<-ch
	}
	return m3
}

func Dot(t1, t2 *Tensor) *Tensor {
	if len(t1.Shape) == 2 && len(t2.Shape) == 2 {
		m1 := t1.Mat
		m2 := t2.Mat
		m3 := dotMat(m1, m2)
		return &Tensor{
			Mat:   m3,
			Shape: []int{m3.Rows, m3.Columns},
		}
	}
	panic([]*Tensor{t1, t2})
}

func IsTheSameShape(t1, t2 *Tensor) bool {
	if len(t1.Shape) == 2 && len(t2.Shape) == 2 {
		m1 := t1.Mat
		m2 := t2.Mat
		return isTheSameShapeMat(m1, m2)
	}
	return false
}

func isTheSameShapeMat(m1, m2 *types.Matrix) bool {
	if m1.Columns == m2.Columns && m1.Rows == m2.Rows {
		return true
	}
	return false
}
