package treeagent

import (
	"runtime"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// BuildTree builds a tree which tries to match the action
// distributions of the samples.
func BuildTree(data []Sample, actionSpace anyrl.LogProber, numFeatures,
	maxDepth int) *Tree {
	res := buildTree(gradientSamples(data, actionSpace), numFeatures, maxDepth)
	res.scaleParams(1 / float64(len(data)))
	return res
}

func buildTree(data []*gradientSample, numFeatures, maxDepth int) *Tree {
	if len(data) == 0 {
		panic("cannot build tree with no data")
	} else if maxDepth == 0 || len(data) == 1 {
		return &Tree{
			Leaf:   true,
			Params: vecToFloats(sumGradients(data)),
		}
	}

	featureChan := make(chan int, numFeatures)
	for i := 0; i < numFeatures; i++ {
		featureChan <- i
	}
	close(featureChan)

	splitChan := make(chan *splitInfo, numFeatures)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			for feature := range featureChan {
				splitChan <- optimalSplit(data, feature)
			}
			wg.Done()
		}()
	}

	go func() {
		wg.Wait()
		close(splitChan)
	}()

	var bestSplit *splitInfo
	for split := range splitChan {
		bestSplit = betterSplit(bestSplit, split)
	}

	if bestSplit == nil {
		// If no split can help, create a leaf.
		return buildTree(data, numFeatures, 0)
	}

	return &Tree{
		Feature:      bestSplit.Feature,
		Threshold:    bestSplit.Threshold,
		LessThan:     buildTree(bestSplit.LeftSamples, numFeatures, maxDepth-1),
		GreaterEqual: buildTree(bestSplit.RightSamples, numFeatures, maxDepth-1),
	}
}

// optimalSplit finds the optimal split for the given
// feature and set of samples.
// It returns nil if no split is effective.
//
// There must be at least one sample.
func optimalSplit(samples []*gradientSample, feature int) *splitInfo {
	sorted, featureVals := sortByFeature(samples, feature)

	rightSum := sumGradients(sorted)
	leftSum := rightSum.Creator().MakeVector(rightSum.Len())
	lastValue := featureVals[0]

	var bestSplit *splitInfo
	for i, sample := range sorted {
		if featureVals[i] > lastValue {
			improvement := splitImprovement(leftSum, rightSum)
			newSplit := &splitInfo{
				Feature:      feature,
				Improvement:  improvement,
				Threshold:    (featureVals[i] + lastValue) / 2,
				LeftSamples:  sorted[:i],
				RightSamples: sorted[i:],
			}
			bestSplit = betterSplit(bestSplit, newSplit)
			lastValue = featureVals[i]
		}
		leftSum.Add(sample.Gradient)
		rightSum.Sub(sample.Gradient)
	}

	return bestSplit
}

func sortByFeature(samples []*gradientSample, feature int) ([]*gradientSample,
	[]float64) {
	var vals []float64
	var sorted []*gradientSample
	for _, sample := range samples {
		vals = append(vals, sample.Feature(feature))
		sorted = append(sorted, sample)
	}

	essentials.VoodooSort(vals, func(i, j int) bool {
		return vals[i] < vals[j]
	}, sorted)

	return sorted, vals
}

func splitImprovement(left, right anyvec.Vector) float64 {
	return numToFloat(left.Dot(left)) + numToFloat(right.Dot(right))
}

type splitInfo struct {
	Feature     int
	Threshold   float64
	Improvement float64

	LeftSamples  []*gradientSample
	RightSamples []*gradientSample
}

func betterSplit(s1, s2 *splitInfo) *splitInfo {
	if s1 == nil {
		return s2
	} else if s2 == nil {
		return s1
	} else if s1.Improvement > s2.Improvement {
		return s1
	} else {
		return s2
	}
}

type gradientSample struct {
	Sample
	Gradient anyvec.Vector
}

func gradientSamples(samples []Sample, space anyrl.LogProber) []*gradientSample {
	res := make([]*gradientSample, len(samples))
	for i, s := range samples {
		params := &anydiff.Var{Vector: s.ActionParams()}
		c := params.Vector.Creator()
		obj := anydiff.Scale(
			space.LogProb(params, s.Action(), 1),
			c.MakeNumeric(s.Advantage()),
		)
		grad := anydiff.NewGrad(params)
		obj.Propagate(anyvec.Ones(c, 1), grad)
		res[i] = &gradientSample{Sample: s, Gradient: grad[params]}
	}
	return res
}

func sumGradients(samples []*gradientSample) anyvec.Vector {
	sum := samples[0].Gradient.Copy()
	for _, sample := range samples[1:] {
		sum.Add(sample.Gradient)
	}
	return sum
}
