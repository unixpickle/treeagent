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

func TestBuildMSE(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	samples := testingSamples(c, 1000)
	builder := &Builder{
		MaxDepth:    2,
		ActionSpace: anyrl.Softmax{},
		Algorithm:   MSEAlgorithm,
	}
	tree := builder.Build(samples)
	if tree.Leaf {
		t.Fatal("expected branching root")
	}
	if tree.LessThan.Leaf || tree.GreaterEqual.Leaf {
		t.Fatal("expected branching children")
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
		t.Fatalf("expected right feature to be 1, but got %d", tree.LessThan.Feature)
	}
}

// testingSamples creates a bunch of samples according to
// a specific set of rules.
// The observation dimensionality is 2, but the first
// component is completely random.
// There are four actions (parameterized via softmax).
// The observation is drawn from [-0.5, 3.5].
// The reward is |actionIdx - observation[1]|.
//
// Multiple calls to testingSamples with the same
// parameters will produce identical samples.
func testingSamples(c anyvec.Creator, numSamples int) []Sample {
	gen := rand.New(rand.NewSource(1337))

	softmax := anyrl.Softmax{}

	var samples []Sample
	for i := 0; i < numSamples; i++ {
		obs := gen.Float64()*4 - 0.5
		outParams := make([]float64, 4)
		for i := range outParams {
			outParams[i] = gen.NormFloat64()
		}
		outVec := c.MakeVectorData(c.MakeNumericList(outParams))
		actionVec := softmax.Sample(outVec, 1)
		actionIdx := anyvec.MaxIndex(actionVec)
		reward := math.Abs(float64(actionIdx) - obs)
		samples = append(samples, &memorySample{
			features:     []float64{gen.NormFloat64(), obs},
			action:       actionVec,
			actionParams: outVec,
			advantage:    reward,
		})
	}

	return samples
}

func BenchmarkBuild(b *testing.B) {
	numFeatures := []int{1000, 10}
	numSamples := []int{100, 5000}
	names := []string{"ManyFeatures", "ManySamples"}
	for i, name := range names {
		b.Run(name, func(b *testing.B) {
			benchmarkBuild(b, numFeatures[i], numSamples[i])
		})
	}
}

func benchmarkBuild(b *testing.B, numFeatures, numSamples int) {
	c := anyvec64.DefaultCreator{}
	samples := benchmarkingSamples(c, numFeatures, numSamples)
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

func benchmarkingSamples(c anyvec.Creator, numFeatures, numSamples int) []Sample {
	var samples []Sample
	for i := 0; i < numSamples; i++ {
		sample := &memorySample{
			features:     make([]float64, numFeatures),
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
	return samples
}
