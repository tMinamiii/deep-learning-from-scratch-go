package nn

func initNetwork() {
	network := map[string]interface{}{}
	network["W1"] = [][]float64{{0.1, 0.3, 0.5}, {0.2, 0.4, 0.6}}
	network["b1"] = []float64{0.1, 0.2, 0.3}
	network["W2"] = [][]float64{{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}}
	network["b2"] = []float64{0.1, 0.2}
	network["W3"] = [][]float64{{0.1, 0.3}, {0.2, 0.4}}
	network["b3"] = []float64{0.1, 0.2}
}

func forward(network map[string]interface{}, x []float64) {
	W1 := network["W1"]
	W2 := network["W2"]
	W3 := network["W3"]
	b1 := network["b1"]
	b2 := network["b2"]
	b3 := network["b3"]
}

// StepFunction is
func StepFunction(x float64) int {
	if x > 0 {
		return 1
	}
	return 0
}
