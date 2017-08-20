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

// A TreeAlgorithm is an algorithm for building trees.
//
// Different tree algorithms solve different objectives.
// Thus, different step sizes may be best for different
// tree algorithms.
type TreeAlgorithm int

const (
	// SumAlgorithm constructs a tree where the leaf nodes
	// contain gradient sums for the repersented samples.
	SumAlgorithm TreeAlgorithm = iota

	// MeanAlgorithm is similar to SumAlgorithm, except
	// that gradients are averaged instead of summed.
	MeanAlgorithm

	// MSEAlgorithm constructs a tree by minimizing
	// mean-squared error over gradients.
	MSEAlgorithm
)

// A Builder builds decision trees to improve on Forest
// policies.
type Builder struct {
	// MaxDepth is the maximum tree depth.
	MaxDepth int

	// ActionSpace is used to determine the probability of
	// actions given the action parameters.
	ActionSpace anyrl.LogProber

	// Regularizer, if non-nil, is used to regularize the
	// action distributions of the policy.
	Regularizer anypg.Regularizer

	// Algorithm specifies how to build the tree.
	Algorithm TreeAlgorithm
}

// Build builds a tree based on the training data.
func (b *Builder) Build(data []Sample) *Tree {
	return b.buildTree(b.gradientSamples(data), b.MaxDepth)
}

func (b *Builder) buildTree(data []*gradientSample, depth int) *Tree {
	if len(data) == 0 {
		panic("cannot build tree with no data")
	} else if depth == 0 || len(data) == 1 {
		res := &Tree{
			Leaf:   true,
			Params: vecToFloats(b.leafParams(data)),
		}
		if b.Algorithm == SumAlgorithm {
			res.scaleParams(1 / float64(len(data)))
		}
		return res
	}

	numFeats := data[0].NumFeatures()

	featureChan := make(chan int, numFeats)
	for i := 0; i < numFeats; i++ {
		featureChan <- i
	}
	close(featureChan)

	splitChan := make(chan *splitInfo, numFeats)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			for feature := range featureChan {
				splitChan <- b.optimalSplit(data, feature)
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
	paramVecs := make([]anyvec.Vector, len(samples))
	outVecs := make([]anyvec.Vector, len(samples))
	advantages := make([]float64, len(samples))
	for i, s := range samples {
		paramVecs[i] = s.ActionParams()
		outVecs[i] = s.Action()
		advantages[i] = s.Advantage()
	}

	c := paramVecs[0].Creator()
	params := &anydiff.Var{Vector: c.Concat(paramVecs...)}
	outs := c.Concat(outVecs...)
	advs := anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(advantages)))
	probs := b.ActionSpace.LogProb(params, outs, len(samples))
	obj := anydiff.Sum(anydiff.Mul(probs, advs))

	if b.Regularizer != nil {
		reg := b.Regularizer.Regularize(params, len(samples))
		obj = anydiff.Add(obj, anydiff.Sum(reg))
	}

	return splitSampleGrads(samples, params, obj)
}

// optimalSplit finds the optimal split for the given
// feature and set of samples.
// It returns nil if no split is effective.
//
// There must be at least one sample.
func (b *Builder) optimalSplit(samples []*gradientSample, feature int) *splitInfo {
	sorted, featureVals := sortByFeature(samples, feature)

	tracker := b.splitTracker()
	tracker.Reset(sorted)
	lastValue := featureVals[0]

	var bestSplit *splitInfo
	for i, sample := range sorted {
		if featureVals[i] > lastValue {
			newSplit := &splitInfo{
				Feature:      feature,
				Threshold:    (featureVals[i] + lastValue) / 2,
				Quality:      tracker.Quality(),
				LeftSamples:  sorted[:i],
				RightSamples: sorted[i:],
			}
			bestSplit = betterSplit(bestSplit, newSplit)
			lastValue = featureVals[i]
		}
		tracker.MoveToLeft(sample)
	}

	return bestSplit
}

func (b *Builder) splitTracker() splitTracker {
	switch b.Algorithm {
	case SumAlgorithm:
		return &sumTracker{}
	case MeanAlgorithm:
		return &meanTracker{}
	case MSEAlgorithm:
		return &mseTracker{}
	default:
		panic("unknown tree algorithm")
	}
}

func (b *Builder) leafParams(data []*gradientSample) anyvec.Vector {
	switch b.Algorithm {
	case SumAlgorithm:
		return sumGradients(data)
	case MeanAlgorithm, MSEAlgorithm:
		sum := sumGradients(data)
		sum.Scale(sum.Creator().MakeNumeric(1 / float64(len(data))))
		return sum
	default:
		panic("unknown tree algorithm")
	}
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

// splitInfo stores information about a feature split.
//
// During training, many potential splitInfos are produced
// and the best ones are selected.
type splitInfo struct {
	Feature   int
	Threshold float64
	Quality   float64

	LeftSamples  []*gradientSample
	RightSamples []*gradientSample
}

// betterSplit selects the better of two splits.
// If a split is nil, the other split is chosen.
func betterSplit(s1, s2 *splitInfo) *splitInfo {
	if s1 == nil {
		return s2
	} else if s2 == nil {
		return s1
	} else if s1.Quality > s2.Quality {
		return s1
	} else {
		return s2
	}
}

type gradientSample struct {
	Sample
	Gradient anyvec.Vector
}

// splitSampleGrads takes the gradient of obj with respect
// to params and splits it up amongst the samples.
func splitSampleGrads(samples []Sample, params *anydiff.Var,
	obj anydiff.Res) []*gradientSample {
	grad := anydiff.NewGrad(params)
	obj.Propagate(anyvec.Ones(params.Output().Creator(), 1), grad)
	gradVec := grad[params]
	gradSize := gradVec.Len() / len(samples)

	res := make([]*gradientSample, len(samples))
	for i, s := range samples {
		res[i] = &gradientSample{
			Sample:   s,
			Gradient: gradVec.Slice(i*gradSize, (i+1)*gradSize),
		}
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

// A splitTracker dynamically computes how good splits are
// on a spectrum of possible splits.
type splitTracker interface {
	Reset(rightSamples []*gradientSample)
	MoveToLeft(sample *gradientSample)
	Quality() float64
}

// A sumTracker is a splitTracker for SumAlgorithm.
type sumTracker struct {
	leftSum  anyvec.Vector
	rightSum anyvec.Vector
}

func (s *sumTracker) Reset(rightSamples []*gradientSample) {
	s.rightSum = sumGradients(rightSamples)
	s.leftSum = s.rightSum.Creator().MakeVector(s.rightSum.Len())
}

func (s *sumTracker) MoveToLeft(sample *gradientSample) {
	s.rightSum.Sub(sample.Gradient)
	s.leftSum.Add(sample.Gradient)
}

func (s *sumTracker) Quality() float64 {
	var sum float64
	for _, vec := range []anyvec.Vector{s.leftSum, s.rightSum} {
		sum += numToFloat(vec.Dot(vec))
	}
	return sum
}

// A meanTracker is a splitTracker for MeanAlgorithm.
type meanTracker struct {
	sumTracker sumTracker
	leftCount  int
	rightCount int
}

func (m *meanTracker) Reset(rightSamples []*gradientSample) {
	m.sumTracker.Reset(rightSamples)
	m.leftCount = 0
	m.rightCount = len(rightSamples)
}

func (m *meanTracker) MoveToLeft(sample *gradientSample) {
	m.sumTracker.MoveToLeft(sample)
	m.leftCount++
	m.rightCount--
}

func (m *meanTracker) Quality() float64 {
	s := &m.sumTracker
	sums := []anyvec.Vector{s.leftSum, s.rightSum}
	counts := []int{m.leftCount, m.rightCount}

	var sum float64
	for i, vec := range sums {
		sum += numToFloat(vec.Dot(vec)) / float64(counts[i])
	}

	return sum
}

// A mseTracker is a splitTracker for MSEAlgorithm.
type mseTracker struct {
	sumTracker   sumTracker
	leftSquares  float64
	rightSquares float64
	leftCount    int
	rightCount   int
}

func (m *mseTracker) Reset(rightSamples []*gradientSample) {
	m.sumTracker.Reset(rightSamples)
	m.leftSquares = 0
	m.rightSquares = 0
	for _, sample := range rightSamples {
		m.rightSquares += numToFloat(sample.Gradient.Dot(sample.Gradient))
	}
	m.leftCount = 0
	m.rightCount = len(rightSamples)
}

func (m *mseTracker) MoveToLeft(sample *gradientSample) {
	m.sumTracker.MoveToLeft(sample)
	sq := numToFloat(sample.Gradient.Dot(sample.Gradient))
	m.leftSquares += sq
	m.rightSquares -= sq
	m.leftCount++
	m.rightCount--
}

func (m *mseTracker) Quality() float64 {
	// The minimal MSE is equivalent to
	//
	//     Var(x) = E[X^2] - E^2[X]
	//
	// Scaling this by n, we get:
	//
	//     Error = (x1^2 + ... + xn^2) - (x1 + ... + xn)^2/n
	//

	sums := []anyvec.Vector{m.sumTracker.leftSum, m.sumTracker.rightSum}
	sqSums := []float64{m.leftSquares, m.rightSquares}
	counts := []int{m.leftCount, m.rightCount}

	var totalError float64
	for i, sum := range sums {
		n := float64(counts[i])
		if n == 0 {
			continue
		}
		totalError += sqSums[i] - numToFloat(sum.Dot(sum))/n
	}
	return -totalError
}
