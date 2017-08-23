package treeagent

import (
	"math"
	"testing"
)

func TestMSETracker(t *testing.T) {
	samples := []*gradientSample{
		{Gradient: []float64{1, 2}},
		{Gradient: []float64{3, 2}},
		{Gradient: []float64{5, 1}},
	}

	tracker := &mseTracker{}
	tracker.Reset(samples)

	// Computed using Octave.
	qualities := []float64{
		-8.66666666666666,
		-2.5,
		-2,
		-8.66666666666666,
	}
	for i := 0; i <= len(samples); i++ {
		actual := tracker.Quality()
		expected := qualities[i]
		if math.Abs(actual-expected) > 1e-5 {
			t.Errorf("split %d: expected %f but got %f", i, expected, actual)
		}
		if i < len(samples) {
			tracker.MoveToLeft(samples[i])
		}
	}
}
