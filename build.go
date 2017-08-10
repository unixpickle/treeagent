package treeagent

import (
	"runtime"
	"sync"

	"github.com/unixpickle/essentials"
)

// BuildTree builds a tree which tries to match the action
// distributions of the samples.
func BuildTree(data []Sample, numFeatures, maxDepth int) *Tree {
	if len(data) == 0 {
		panic("cannot build tree with no data")
	} else if maxDepth == 0 || len(data) == 1 {
		return &Tree{
			Distribution: addActionDists(sampleDists(data)).normalize(),
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
		return BuildTree(data, numFeatures, 0)
	}

	return &Tree{
		Feature:      bestSplit.Feature,
		Threshold:    bestSplit.Threshold,
		LessThan:     BuildTree(bestSplit.LeftSamples, numFeatures, maxDepth-1),
		GreaterEqual: BuildTree(bestSplit.RightSamples, numFeatures, maxDepth-1),
	}
}

// optimalSplit finds the optimal split for the given
// feature and set of samples.
// It returns nil if no split is effective.
//
// There must be at least one sample.
func optimalSplit(samples []Sample, feature int) *splitInfo {
	sorted, featureVals := sortByFeature(samples, feature)
	dists := sampleDists(sorted)

	rightSums := make([]ActionDist, len(samples))
	rightSum := zeroActionDist(len(samples[0].ActionDist()))
	for i := len(samples) - 1; i >= 0; i-- {
		rightSum = rightSum.add(dists[i])
		rightSums[i] = rightSum
	}
	leftSum := zeroActionDist(len(rightSum))
	lastValue := featureVals[0]

	var bestSplit *splitInfo
	for i, dist := range dists {
		if featureVals[i] > lastValue {
			newSplit := &splitInfo{
				Feature:      feature,
				Loss:         lossFromSums(leftSum, rightSums[i]),
				Threshold:    (featureVals[i] + lastValue) / 2,
				LeftSamples:  sorted[:i],
				RightSamples: sorted[i:],
			}
			bestSplit = betterSplit(bestSplit, newSplit)
			lastValue = featureVals[i]
		}
		leftSum = leftSum.add(dist)
	}

	return bestSplit
}

func sortByFeature(samples []Sample, feature int) ([]Sample, []float64) {
	var vals []float64
	var sorted []Sample
	for _, sample := range samples {
		vals = append(vals, sample.Feature(feature))
		sorted = append(sorted, sample)
	}

	essentials.VoodooSort(vals, func(i, j int) bool {
		return vals[i] < vals[j]
	}, sorted)

	return sorted, vals
}

func lossFromSums(leftSum, rightSum ActionDist) float64 {
	return lossFromDistSum(leftSum) + lossFromDistSum(rightSum)
}

func lossFromDistSum(sum ActionDist) float64 {
	return -sum.normalize().dot(sum.exp())
}

func sampleDists(samples []Sample) []ActionDist {
	res := make([]ActionDist, len(samples))
	for i, x := range samples {
		res[i] = x.ActionDist()
	}
	return res
}

type splitInfo struct {
	Feature   int
	Threshold float64
	Loss      float64

	LeftSamples  []Sample
	RightSamples []Sample
}

func betterSplit(s1, s2 *splitInfo) *splitInfo {
	if s1 == nil {
		return s2
	} else if s2 == nil {
		return s1
	} else if s1.Loss < s2.Loss {
		return s1
	} else {
		return s2
	}
}
