package treeagent

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec64"
)

const (
	numBenchmarkFeatures = 1000
	numBenchmarkSamples  = 100
	benchmarkDepth       = 3
)

func TestMSETracker(t *testing.T) {
	c := anyvec64.DefaultCreator{}

	samples := []*gradientSample{
		{Gradient: c.MakeVectorData([]float64{1, 2})},
		{Gradient: c.MakeVectorData([]float64{3, 2})},
		{Gradient: c.MakeVectorData([]float64{5, 1})},
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

func BenchmarkBuild(b *testing.B) {
	c := anyvec64.DefaultCreator{}
	var samples []Sample
	for i := 0; i < numBenchmarkSamples; i++ {
		sample := &memorySample{
			features:     make([]float64, numBenchmarkFeatures),
			action:       c.MakeVector(2),
			actionParams: c.MakeVector(2),
			advantage:    rand.NormFloat64(),
		}
		for i := range sample.features {
			sample.features[i] = rand.NormFloat64()
		}
		idx := rand.Intn(2)
		sample.action.Slice(idx, idx+1).AddScalar(1.0)
		samples = append(samples, sample)
	}

	builder := &Builder{
		MaxDepth:    benchmarkDepth,
		ActionSpace: anyrl.Softmax{},
	}

	for _, multithread := range []bool{false, true} {
		name := "Single"
		if multithread {
			name = "Multi"
		}
		b.Run(name, func(b *testing.B) {
			if !multithread {
				old := runtime.GOMAXPROCS(0)
				runtime.GOMAXPROCS(1)
				defer runtime.GOMAXPROCS(old)
			}
			for i := 0; i < b.N; i++ {
				builder.Build(samples)
			}
		})
	}
}
