package treeagent

import (
	"math"
	"math/rand"
	"testing"
)

func TestBuildTree(t *testing.T) {
	// If the second feature is less than 2, then
	// action 0 is better.
	// Otherwise, action 1 is better.
	samples := []Sample{
		&memorySample{features: []float64{3, 0.5}, action: 0, advantage: 3},
		&memorySample{features: []float64{3, 1}, action: 0, advantage: 5},
		&memorySample{features: []float64{3, 1.5}, action: 1, advantage: -5},
		&memorySample{features: []float64{3, 1.75}, action: 1, advantage: -3},

		&memorySample{features: []float64{3, 2.25}, action: 1, advantage: 5},
		&memorySample{features: []float64{3, 2.5}, action: 1, advantage: 3},
		&memorySample{features: []float64{3, 3.1}, action: 0, advantage: -0.3},
		&memorySample{features: []float64{3, 3.7}, action: 0, advantage: -20},
	}
	for i := 0; i < len(samples)-1; i++ {
		j := rand.Intn(len(samples)-i) + i
		samples[i], samples[j] = samples[j], samples[i]
	}

	tree := BuildTree(samples, 2, 1)
	if tree.Leaf {
		t.Fatal("expected non-leaf")
	}
	if tree.Feature != 1 {
		t.Error("expected feature 0")
	}
	if math.Abs(tree.Threshold-2) > 1e-5 {
		t.Errorf("expected threshold of 2 but got %f", tree.Threshold)
	}
	if tree.LessThan.Action != 0 {
		t.Error("bad less-than action")
	}
	if tree.GreaterEqual.Action != 1 {
		t.Error("bad greater-equal action")
	}
}
