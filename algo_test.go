package treeagent

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/unixpickle/essentials"
)

func TestMSETracker(t *testing.T) {
	testTrackersEquivalent(t, &mseTracker{}, &naiveMSETracker{})
}

func TestStddevTracker(t *testing.T) {
	testTrackersEquivalent(t, &stddevTracker{}, &naiveStddevTracker{})
}

func testTrackersEquivalent(t *testing.T, t1, t2 splitTracker) {
	samples := make([]*gradientSample, 100)
	for i := range samples {
		samples[i] = &gradientSample{Gradient: make([]float64, 5)}
		for j := range samples[i].Gradient {
			samples[i].Gradient[j] = rand.NormFloat64()
		}
	}

	var qualities [2][]float64
	var orders [2][]int
	for i, tracker := range []splitTracker{t1, t2} {
		tracker.Reset(samples)
		tracker.MoveToLeft(samples[0])
		for j, sample := range samples[1:] {
			qualities[i] = append(qualities[i], tracker.Quality())
			orders[i] = append(orders[i], j)
			tracker.MoveToLeft(sample)
		}
		essentials.VoodooSort(qualities[i], func(j, k int) bool {
			return qualities[i][j] < qualities[i][k]
		}, orders[i])
	}

	if !reflect.DeepEqual(orders[0], orders[1]) {
		t.Error("got different orderings")
	}
}

type naiveMSETracker struct {
	Left  []*gradientSample
	Right []*gradientSample
}

func (n *naiveMSETracker) Reset(right []*gradientSample) {
	n.Left = right[:0]
	n.Right = right
}

func (n *naiveMSETracker) MoveToLeft(sample *gradientSample) {
	n.Left = n.Left[:len(n.Left)+1]
	n.Right = n.Right[1:]
}

func (n *naiveMSETracker) Quality() float64 {
	return -(naiveMSE(n.Left) + naiveMSE(n.Right))
}

type naiveStddevTracker struct {
	naiveMSETracker
}

func (n *naiveStddevTracker) Quality() float64 {
	return -(naiveWeightedStddev(n.Left) + naiveWeightedStddev(n.Right))
}

func naiveWeightedStddev(samples []*gradientSample) float64 {
	variance := naiveMSE(samples) / float64(len(samples))
	return float64(len(samples)) * math.Sqrt(variance)
}

func naiveMSE(samples []*gradientSample) float64 {
	mean := sumGradients(samples).Scale(1 / float64(len(samples)))
	var mse float64
	for _, sample := range samples {
		diff := mean.Copy().Sub(sample.Gradient)
		mse += diff.Dot(diff)
	}
	return mse
}
