package treeagent

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestPPOMSE(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	base := testingRandomForest()
	samples := testingSamples(c, 5000, base)
	ppo := &PPO{
		Builder: &Builder{
			MaxDepth:    2,
			ActionSpace: anyrl.Softmax{},
			Algorithm:   MSEAlgorithm,
		},
	}
	tree, _ := ppo.Step(samples, base)
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
