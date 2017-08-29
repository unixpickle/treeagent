package treeagent

import (
	"math/rand"
	"runtime"
	"testing"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestPPOMSE(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	base := testingRandomForest()
	samples := testingSamples(c, 5000, base)
	ppo := &PPO{
		PG: PG{
			Builder: Builder{
				Algorithm: MSEAlgorithm,
				MaxDepth:  2,
			},
			ActionSpace: anyrl.Softmax{},
		},
	}
	tree, _, _ := ppo.Build(samples, base)
	verifyTestingSamplesTree(t, tree)
}

// testingRandomForest generates a Forest which is
// compatible with testingSamples.
func testingRandomForest() *Forest {
	gen := rand.New(rand.NewSource(1337))
	res := NewForest(4)
	for i := 0; i < 10; i++ {
		params := make([]ActionParams, 2)
		for j := range params {
			params[j] = make(ActionParams, 4)
			for k := range params[j] {
				params[j][k] = gen.NormFloat64()
			}
		}

		thresh := gen.NormFloat64()
		res.Trees = append(res.Trees, &Tree{
			Feature:      0,
			Threshold:    thresh,
			LessThan:     &Tree{Leaf: true, Params: params[0]},
			GreaterEqual: &Tree{Leaf: true, Params: params[1]},
		})
		res.Weights = append(res.Weights, 0.1)
	}
	return res
}

func BenchmarkPPO(b *testing.B) {
	numFeatures := []int{1000, 10}
	numSamples := []int{100, 5000}
	names := []string{"ManyFeatures", "ManySamples"}
	for i, name := range names {
		b.Run(name, func(b *testing.B) {
			benchmarkPPO(b, numFeatures[i], numSamples[i])
		})
	}
}

func benchmarkPPO(b *testing.B, numFeatures, numSamples int) {
	c := anyvec64.DefaultCreator{}
	samples := benchmarkingSamples(c, numFeatures, numSamples, false)
	ppo := &PPO{
		PG: PG{
			Builder: Builder{
				MaxDepth: benchmarkDepth,
			},
			ActionSpace: anyrl.Softmax{},
		},
	}
	base := NewForest(2)

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
				ppo.Build(samples, base)
			}
		})
	}
}
