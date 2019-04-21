package logic

import (
	"testing"
)

func TestAND_0_0(t *testing.T) {
	if AND(0, 0) != 0 {
		t.Fail()
	}
}
func TestAND_0_1(t *testing.T) {
	if AND(0, 1) != 0 {
		t.Fail()
	}
}
func TestAND_1_0(t *testing.T) {
	if AND(1, 0) != 0 {
		t.Fail()
	}
}
func TestAND_1_1(t *testing.T) {
	if AND(1, 1) != 1 {
		t.Fail()
	}
}

func TestNAND_0_0(t *testing.T) {
	if NAND(0, 0) != 1 {
		t.Fail()
	}
}
func TestNAND_0_1(t *testing.T) {
	if NAND(0, 1) != 1 {
		t.Fail()
	}
}
func TestNAND_1_0(t *testing.T) {
	if NAND(1, 0) != 1 {
		t.Fail()
	}
}
func TestNAND_1_1(t *testing.T) {
	if NAND(1, 1) != 0 {
		t.Fail()
	}
}

func TestOR_0_0(t *testing.T) {
	if OR(0, 0) != 0 {
		t.Fail()
	}
}
func TestOR_0_1(t *testing.T) {
	if OR(0, 1) != 1 {
		t.Fail()
	}
}
func TestOR_1_0(t *testing.T) {
	if OR(1, 0) != 1 {
		t.Fail()
	}
}
func TestOR_1_1(t *testing.T) {
	if OR(1, 1) != 1 {
		t.Fail()
	}
}

func TestXOR_0_0(t *testing.T) {
	if XOR(0, 0) != 0 {
		t.Fail()
	}
}
func TestXOR_0_1(t *testing.T) {
	if XOR(0, 1) != 1 {
		t.Fail()
	}
}
func TestXOR_1_0(t *testing.T) {
	if XOR(1, 0) != 1 {
		t.Fail()
	}
}
func TestXOR_1_1(t *testing.T) {
	if XOR(1, 1) != 0 {
		t.Fail()
	}
}
