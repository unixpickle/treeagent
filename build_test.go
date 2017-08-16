package treeagent

import (
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
				BuildTree(samples, anyrl.Softmax{}, numBenchmarkFeatures, benchmarkDepth)
			}
		})
	}

}
