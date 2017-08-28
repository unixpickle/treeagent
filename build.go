package treeagent

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/essentials"
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

	// FeatureFrac is the fraction of features to try for
	// each branching node.
	//
	// If 0, all features are tried.
	FeatureFrac float64

	// MinLeaf is the minimum number of representative
	// samples for a leaf node.
	// A split will never occur such that either of the
	// two branches gets fewer than MinLeaf samples.
	MinLeaf int

	// ParamWhitelist, if non-nil, specifies the parameter
	// indices to target with the trees.
	// All parameters which are not specified will be 0.
	ParamWhitelist []int
}

// Build builds a tree based on the training data.
func (b *Builder) Build(data []Sample) *Tree {
	gradData := b.gradientSamples(data)
	return b.buildTree(gradData, gradData, b.MaxDepth)
}

func (b *Builder) buildTree(data, allData []*gradientSample, depth int) *Tree {
	if len(data) == 0 {
		panic("cannot build tree with no data")
	} else if depth == 0 || len(data) == 1 {
		res := &Tree{
			Leaf:   true,
			Params: ActionParams(b.Algorithm.leafParams(data, allData)),
		}
		if b.Algorithm == SumAlgorithm || b.Algorithm == BalancedSumAlgorithm {
			res.scaleParams(1 / float64(len(data)))
		} else if b.Algorithm == SignAlgorithm {
			res = SignTree(res)
		}
		return res
	}

	numFeatures := data[0].NumFeatures()
	featureChan := b.featuresToTry(numFeatures)
	splitChan := make(chan *splitInfo, numFeatures)

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
		return b.buildTree(data, allData, 0)
	}

	return &Tree{
		Feature:      bestSplit.Feature,
		Threshold:    bestSplit.Threshold,
		LessThan:     b.buildTree(bestSplit.LeftSamples, allData, depth-1),
		GreaterEqual: b.buildTree(bestSplit.RightSamples, allData, depth-1),
	}
}

func (b *Builder) gradientSamples(samples []Sample) []*gradientSample {
	_, res := computeObjective(samples, nil, b.objective)
	return b.maskGradients(res)
}

// objective implements the policy gradient objective
// function with optional regularization.
func (b *Builder) objective(params, old, acts, advs anydiff.Res, n int) anydiff.Res {
	probs := b.ActionSpace.LogProb(params, acts.Output(), n)
	obj := anydiff.Sum(anydiff.Mul(probs, advs))

	if b.Regularizer != nil {
		reg := b.Regularizer.Regularize(params, n)
		obj = anydiff.Add(obj, anydiff.Sum(reg))
	}

	return obj
}

// optimalSplit finds the optimal split for the given
// feature and set of samples.
// It returns nil if no split is effective.
//
// There must be at least one sample.
func (b *Builder) optimalSplit(samples []*gradientSample, feature int) *splitInfo {
	sorted, featureVals := sortByFeature(samples, feature)

	tracker := b.Algorithm.splitTracker()
	tracker.Reset(sorted)
	lastValue := featureVals[0]

	var bestSplit *splitInfo
	for i, sample := range sorted {
		if featureVals[i] > lastValue {
			if i >= b.MinLeaf && len(samples)-i >= b.MinLeaf {
				newSplit := &splitInfo{
					Feature:      feature,
					Threshold:    (featureVals[i] + lastValue) / 2,
					Quality:      tracker.Quality(),
					LeftSamples:  sorted[:i],
					RightSamples: sorted[i:],
				}
				bestSplit = betterSplit(bestSplit, newSplit)
			}
			lastValue = featureVals[i]
		}
		tracker.MoveToLeft(sample)
	}

	return bestSplit
}

func (b *Builder) featuresToTry(numFeatures int) <-chan int {
	useFeatures := numFeatures
	if b.FeatureFrac != 0 {
		if b.FeatureFrac < 0 || b.FeatureFrac > 1 {
			panic("feature fraction out of range")
		}
		useFeatures = int(math.Ceil(b.FeatureFrac * float64(numFeatures)))
	}
	featureChan := make(chan int, useFeatures)
	if useFeatures != numFeatures {
		for _, i := range rand.Perm(numFeatures)[:useFeatures] {
			featureChan <- i
		}
	} else {
		for i := 0; i < numFeatures; i++ {
			featureChan <- i
		}
	}
	close(featureChan)
	return featureChan
}

func (b *Builder) maskGradients(samples []*gradientSample) []*gradientSample {
	if len(samples) == 0 || b.ParamWhitelist == nil {
		return samples
	}
	mask := make(smallVec, len(samples[0].Gradient))
	for _, idx := range b.ParamWhitelist {
		mask[idx] = 1
	}
	for _, sample := range samples {
		sample.Gradient.Mul(mask)
	}
	return samples
}

func sortByFeature(samples []*gradientSample, feature int) ([]*gradientSample,
	[]float64) {
	vals := make([]float64, len(samples))
	sorted := make([]*gradientSample, len(samples))
	for i, sample := range samples {
		vals[i] = sample.Feature(feature)
		sorted[i] = sample
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
