package treeagent

import (
	"runtime"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// A Builder builds decision trees to improve on Forest
// policies.
type Builder struct {
	// NumFeatures is the number of observation features.
	NumFeatures int

	// MaxDepth is the maximum tree depth.
	MaxDepth int

	// ActionSpace is used to determine the probability of
	// actions given the action parameters.
	ActionSpace anyrl.LogProber

	// Regularizer, if non-nil, is used to regularize the
	// action distributions of the policy.
	Regularizer anypg.Regularizer
}

// Build builds a tree based on the training data.
func (b *Builder) Build(data []Sample) *Tree {
	res := b.buildTree(b.gradientSamples(data), b.MaxDepth)
	res.scaleParams(1 / float64(len(data)))
	return res
}

func (b *Builder) buildTree(data []*gradientSample, depth int) *Tree {
	if len(data) == 0 {
		panic("cannot build tree with no data")
	} else if depth == 0 || len(data) == 1 {
		return &Tree{
			Leaf:   true,
			Params: vecToFloats(sumGradients(data)),
		}
	}

	featureChan := make(chan int, b.NumFeatures)
	for i := 0; i < b.NumFeatures; i++ {
		featureChan <- i
	}
	close(featureChan)

	splitChan := make(chan *splitInfo, b.NumFeatures)

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
		return b.buildTree(data, 0)
	}

	return &Tree{
		Feature:      bestSplit.Feature,
		Threshold:    bestSplit.Threshold,
		LessThan:     b.buildTree(bestSplit.LeftSamples, depth-1),
		GreaterEqual: b.buildTree(bestSplit.RightSamples, depth-1),
	}
}

func (b *Builder) gradientSamples(samples []Sample) []*gradientSample {
	res := make([]*gradientSample, len(samples))
	for i, s := range samples {
		params := &anydiff.Var{Vector: s.ActionParams()}
		c := params.Vector.Creator()
		obj := anydiff.Scale(
			b.ActionSpace.LogProb(params, s.Action(), 1),
			c.MakeNumeric(s.Advantage()),
		)
		if b.Regularizer != nil {
			obj = anydiff.Add(obj, b.Regularizer.Regularize(params, 1))
		}
		grad := anydiff.NewGrad(params)
		obj.Propagate(anyvec.Ones(c, 1), grad)
		res[i] = &gradientSample{Sample: s, Gradient: grad[params]}
	}
	return res
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

func sumGradients(samples []*gradientSample) anyvec.Vector {
	sum := samples[0].Gradient.Copy()
	for _, sample := range samples[1:] {
		sum.Add(sample.Gradient)
	}
	return sum
}
