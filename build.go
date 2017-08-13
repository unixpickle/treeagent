package treeagent

import (
	"math"
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
			Leaf:   true,
			Action: bestAction(data),
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

	rightSum := newRewardAverages()
	leftSum := newRewardAverages()
	for _, sample := range samples {
		rightSum.Add(sample)
	}
	lastValue := featureVals[0]

	var bestSplit *splitInfo
	for i, sample := range sorted {
		if featureVals[i] > lastValue {
			newSplit := &splitInfo{
				Feature:      feature,
				Advantage:    advantageForSplit(leftSum, rightSum),
				Threshold:    (featureVals[i] + lastValue) / 2,
				LeftSamples:  sorted[:i],
				RightSamples: sorted[i:],
			}
			bestSplit = betterSplit(bestSplit, newSplit)
			lastValue = featureVals[i]
		}
		leftSum.Add(sample)
		rightSum.Remove(sample)
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

type splitInfo struct {
	Feature   int
	Threshold float64
	Advantage float64

	LeftSamples  []Sample
	RightSamples []Sample
}

func advantageForSplit(left, right *rewardAverages) float64 {
	return left.GreedyAdvantage() + right.GreedyAdvantage()
}

func betterSplit(s1, s2 *splitInfo) *splitInfo {
	if s1 == nil {
		return s2
	} else if s2 == nil {
		return s1
	} else if s1.Advantage > s2.Advantage {
		return s1
	} else {
		return s2
	}
}

func bestAction(data []Sample) int {
	avg := newRewardAverages()
	for _, sample := range data {
		avg.Add(sample)
	}
	return avg.BestAction()
}

type rewardAverages struct {
	ActionTotals map[int]float64
	ActionCounts map[int]int
	Count        int
}

func newRewardAverages() *rewardAverages {
	return &rewardAverages{
		ActionTotals: map[int]float64{},
		ActionCounts: map[int]int{},
	}
}

func (r *rewardAverages) Add(sample Sample) {
	r.ActionTotals[sample.Action()] += sample.Advantage()
	r.ActionCounts[sample.Action()]++
	r.Count++
}

func (r *rewardAverages) Remove(sample Sample) {
	r.ActionTotals[sample.Action()] -= sample.Advantage()
	r.ActionCounts[sample.Action()]--
	r.Count--
}

func (r *rewardAverages) BestAction() int {
	var bestAction int
	bestAdvantage := math.Inf(-1)
	for action, total := range r.ActionTotals {
		mean := total / float64(r.ActionCounts[action])
		if mean >= bestAdvantage {
			bestAdvantage = mean
			bestAction = action
		}
	}
	return bestAction
}

func (r *rewardAverages) ActionMean(action int) float64 {
	return r.ActionTotals[action] / float64(r.ActionCounts[action])
}

func (r *rewardAverages) GreedyAdvantage() float64 {
	return r.ActionTotals[r.BestAction()]
}
