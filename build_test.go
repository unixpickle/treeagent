package treeagent

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

const (
	benchmarkDepth = 3
)

func TestBuildMSE(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	samples := testingSamples(c, 5000, nil)
	builder := &Builder{
		MaxDepth:    2,
		ActionSpace: anyrl.Softmax{},
		Algorithm:   MSEAlgorithm,
	}
	tree := builder.Build(samples)
	verifyTestingSamplesTree(t, tree)
}

// testingSamples creates a bunch of samples according to
// a specific set of rules.
// The observation dimensionality is 2, but the first
// component is completely random.
// There are four actions (parameterized via softmax).
// The value of observation[1] is drawn from [-0.5, 3.5].
//
// The reward is 1 if |actionIdx - observation[1]| <= 0.5,
// or 0.5 if |actionIdx - observation[1]| <= 1.5.
// Otherwise, the reward is 0.
//
// Multiple calls to testingSamples with the same
// parameters will produce identical samples.
//
// If f is non-nil, then it is used to sample actions.
// Otherwise, softmax parameters are generated from the
// normal distribution.
func testingSamples(c anyvec.Creator, numSamples int, f *Forest) []Sample {
	gen := rand.New(rand.NewSource(1337))

	softmax := anyrl.Softmax{}

	var samples []Sample
	for i := 0; i < numSamples; i++ {
		obs := gen.Float64()*4 - 0.5
		features := []float64{gen.NormFloat64(), obs}

		outParams := make([]float64, 4)
		if f == nil {
			for i := range outParams {
				outParams[i] = gen.NormFloat64()
			}
		} else {
			copy(outParams, f.Apply(features))
		}

		outVec := c.MakeVectorData(c.MakeNumericList(outParams))
		actionVec := softmax.Sample(outVec, 1)
		actionIdx := anyvec.MaxIndex(actionVec)
		var reward float64
		diff := math.Abs(float64(actionIdx) - obs)
		if diff <= 0.5 {
			reward = 1
		} else if diff <= 1.5 {
			reward = 0.5
		}
		samples = append(samples, &memorySample{
			features:     features,
			action:       actionVec,
			actionParams: outVec,
			advantage:    reward,
		})
	}

	return samples
}

// verifyTestingSamplesTree checks that a depth-2 tree
// efficiently captures the solution to the samples from
// testingSamples.
func verifyTestingSamplesTree(t *testing.T, tree *Tree) {
	if tree.Leaf {
		t.Error("expected branching root")
		return
	}
	if tree.LessThan.Leaf || tree.GreaterEqual.Leaf {
		t.Error("expected branching children")
		return
	}

	if tree.Feature == 1 {
		if math.Abs(tree.Threshold-1.5) > 0.5 {
			t.Errorf("expected root threshold around 1.5, but got %f", tree.Threshold)
		}
	} else {
		t.Errorf("expected root feature to be 1, but got %d", tree.Feature)
	}

	if tree.LessThan.Feature == 1 {
		if math.Abs(tree.LessThan.Threshold-0.5) > 0.5 {
			t.Errorf("expected left threshold around 0.5, but got %f",
				tree.LessThan.Threshold)
		}
	} else {
		t.Errorf("expected left feature to be 1, but got %d", tree.LessThan.Feature)
	}

	if tree.GreaterEqual.Feature == 1 {
		if math.Abs(tree.GreaterEqual.Threshold-2.5) > 0.5 {
			t.Errorf("expected right threshold around 2.5, but got %f",
				tree.GreaterEqual.Threshold)
		}
	} else {
		t.Errorf("expected right feature to be 1, but got %d", tree.LessThan.Feature)
		return
	}
}

func BenchmarkBuild(b *testing.B) {
	numFeatures := []int{1000, 10}
	numSamples := []int{100, 5000}
	names := []string{"ManyFeatures", "ManySamples"}
	byteObs := []bool{false, false}

	if !testing.Short() {
		// Simulate a batch of Pong frames.
		numFeatures = append(numFeatures, 9600)
		numSamples = append(numSamples, 4096)
		byteObs = append(byteObs, true)
		names = append(names, "Huge")
	}

	for i, name := range names {
		b.Run(name, func(b *testing.B) {
			benchmarkBuild(b, numFeatures[i], numSamples[i], byteObs[i])
		})
	}
}

func benchmarkBuild(b *testing.B, numFeatures, numSamples int, byteObs bool) {
	c := anyvec64.DefaultCreator{}
	samples := benchmarkingSamples(c, numFeatures, numSamples, byteObs)
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

func benchmarkingSamples(c anyvec.Creator, numFeatures, numSamples int,
	byteObs bool) []Sample {
	var samples []Sample
	for i := 0; i < numSamples; i++ {
		sample := &memorySample{
			features:     make([]float64, numFeatures),
			action:       c.MakeVector(2),
			actionParams: c.MakeVector(2),
			advantage:    rand.NormFloat64(),
		}
		for i := range sample.features {
			if byteObs {
				sample.features[i] = float64(rand.Intn(0x100))
			} else {
				sample.features[i] = rand.NormFloat64()
			}
		}
		idx := rand.Intn(2)
		sample.action.Slice(idx, idx+1).AddScalar(1.0)
		samples = append(samples, sample)
	}

	if byteObs {
		for i, sample := range samples {
			samples[i] = newUint8Sample(sample)
		}
	}

	return samples
}
