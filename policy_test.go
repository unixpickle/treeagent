package treeagent

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/approb"
	"github.com/unixpickle/weakai/idtrees"
)

func TestEpsilonSampled(t *testing.T) {
	dist := testClassifier{
		0: 0.3,
		1: 0.5,
		2: 0.15,
		3: 0.05,
	}
	policy := &Policy{
		Classifier: dist,
		NumActions: 4,
		Epsilon:    0.3,
	}
	corr := approb.Correlation(50000, 0.3, func() float64 {
		if rand.Float64() < 0.3 {
			return float64(rand.Intn(4))
		}
		return float64(sampleTestClassifier(dist))
	}, func() float64 {
		return float64(sampleTestClassifier(policy))
	})
	if corr < 0.9999 {
		t.Error("correlation should be near 1, but got", corr)
	}
}

func TestEpsilonGreedy(t *testing.T) {
	dist := testClassifier{
		0: 0.3,
		1: 0.5,
		2: 0.15,
		3: 0.05,
	}
	policy := &Policy{
		Classifier: dist,
		NumActions: 4,
		Epsilon:    0.3,
		Greedy:     true,
	}
	corr := approb.Correlation(50000, 0.3, func() float64 {
		if rand.Float64() < 0.3 {
			return float64(rand.Intn(4))
		} else {
			return 1
		}
	}, func() float64 {
		return float64(sampleTestClassifier(policy))
	})
	if corr < 0.9999 {
		t.Error("correlation should be near 1, but got", corr)
	}
}

func sampleTestClassifier(c Classifier) int {
	out := c.Classify(nil)
	off := rand.Float64()
	for i := 0; i < 4; i++ {
		off -= out[i]
		if off <= 0 {
			return i
		}
	}
	return 3
}

type testClassifier map[idtrees.Class]float64

func (t testClassifier) Classify(s idtrees.AttrMap) map[idtrees.Class]float64 {
	return t
}
