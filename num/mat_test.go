package num

import (
	"log"
	"testing"
)

func TestMul_1(t *testing.T) {
	m1, _ := NewMat64(1, 2, []float64{1, 2})
	m2, _ := NewMat64(2, 1, []float64{3, 4})
	r, _ := Mul(m1, m2)
	if !ArrayEqual(r.Array, []float64{11}) {
		log.Println(r.Array)
		t.Fail()
	}
}

func TestMul_2(t *testing.T) {
	m1, _ := NewMat64(1, 2, []float64{1, 2})
	m2, _ := NewMat64(2, 2, []float64{
		1, 2,
		3, 4,
	})
	r, _ := Mul(m1, m2)
	if !ArrayEqual(r.Array, []float64{7, 10}) {
		log.Println(r.Array)
		t.Fail()
	}
}


func TestMul_3(t *testing.T) {
	m1, _ := NewMat64(2, 2, []float64{
		1, 2,
		3, 4,
	})
	m2, _ := NewMat64(2, 2, []float64{
		1, 0,
		0, 1,
	})
	r, _ := Mul(m1, m2)
	if !ArrayEqual(r.Array, []float64{
		1, 2,
		3, 4,
	}) {
		log.Println(r.Array)
		t.Fail()
	}
}

func TestMul_4(t *testing.T) {
	m1, _ := NewMat64(1, 2, []float64{ 1, 2 })
	m2, _ := NewMat64(2, 3, []float64{
		1, 1, 1,
		2, 2, 2,
	})
	r, _ := Mul(m1, m2)
	if !ArrayEqual(r.Array, []float64{5, 5, 5}) {
		log.Println(r.Array)
		t.Fail()
	}
}
