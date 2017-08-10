package treeagent

import (
	"math"
	"testing"
)

func TestBuildTreeSplit(t *testing.T) {
	samples := []*testingSample{
		{
			Features: []float64{3, 0.5},
			Dist:     map[Action]float64{0: 0.7, 1: 0.3},
		},
		{
			Features: []float64{3, 2},
			Dist:     map[Action]float64{0: 0.5, 1: 0.5},
		},
		{
			Features: []float64{3, 1.5},
			Dist:     map[Action]float64{0: 0.7, 1: 0.3},
		},
		{
			Features: []float64{3, 2.5},
			Dist:     map[Action]float64{0: 0.5, 1: 0.5},
		},
		{
			Features: []float64{3, 1},
			Dist:     map[Action]float64{0: 0.7, 1: 0.3},
		},
		{
			Features: []float64{3, 2.5},
			Dist:     map[Action]float64{0: 0.5, 1: 0.5},
		},
	}
	tree := buildTestTree(samples, 2, 1)
	if tree.Feature != 1 || math.Abs(tree.Threshold-1.75) > 1e-4 {
		t.Errorf("expected (1,1.75) but got (%d,%f)", tree.Feature, tree.Threshold)
	}
}

func buildTestTree(samples []*testingSample, numFeatures, maxDepth int) *Tree {
	interfaces := make([]Sample, len(samples))
	for i, s := range samples {
		interfaces[i] = s
	}
	return BuildTree(interfaces, numFeatures, maxDepth)
}

type testingSample struct {
	Features []float64
	Dist     ActionDist
}

func (t *testingSample) Feature(idx int) float64 {
	return t.Features[idx]
}

func (t *testingSample) ActionDist() ActionDist {
	return t.Dist
}
