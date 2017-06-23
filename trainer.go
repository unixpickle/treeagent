package treeagent

import (
	"math"
	"math/rand"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/weakai/idtrees"
)

// A Trainer builds trees using a hill-climbing algorithm.
type Trainer struct {
	// NumTrees is the number of trees in the forest.
	NumTrees int

	// NumFeatures is the number of input features.
	NumFeatures int

	// RolloutFrac is the fraction of rollouts to clone.
	// A value closer to 0 encourages more aggressive
	// hill-climbing.
	//
	// If 0, a default of 0.5 is used.
	RolloutFrac float64

	// UseFeatures is the number of features to use for
	// each tree in the forest.
	//
	// If 0, NumFeatures is used.
	UseFeatures int

	// UseSteps returns the fraction of timesteps to use
	// as training samples for each tree, given the total
	// number of timesteps.
	//
	// If nil, all timesteps are used as data.
	UseSteps func(total int) float64

	// BuildTree builds a tree for the dataset.
	//
	// If nil, ID3 is used.
	BuildTree func(samples []idtrees.Sample, attrs []idtrees.Attr) *idtrees.Tree
}

// Train trains a random forest on the rollouts.
func (t *Trainer) Train(r *anyrl.RolloutSet) idtrees.Forest {
	var res idtrees.Forest
	for i := 0; i < t.NumTrees; i++ {
		tree := t.sampleTree(r)
		if tree != nil {
			res = append(res, tree)
		}
	}
	return res
}

func (t *Trainer) sampleTree(r *anyrl.RolloutSet) *idtrees.Tree {
	indices := t.bestRolloutIndices(r)

	// Use a subset of the features to train this tree.
	features := []idtrees.Attr{}
	mapping := map[idtrees.Attr]idtrees.Attr{}
	featurePerm := rand.Perm(t.NumFeatures)[:t.useFeatures()]
	for i, j := range featurePerm {
		mapping[j] = i
		features = append(features, j)
	}

	totalSamples := 0
	for _, idx := range indices {
		totalSamples += len(r.Rewards[idx])
	}
	sampleProb := t.useSteps(totalSamples)

	var samples []idtrees.Sample
	actions := r.Actions.ReadTape(0, -1)
	for input := range r.Inputs.ReadTape(0, -1) {
		split := splitBatch(input)
		splitActions := splitBatch(<-actions)
		for _, idx := range indices {
			if !input.Present[idx] || rand.Float64() > sampleProb {
				continue
			}
			inputVec := split[idx]
			action := anyvec.MaxIndex(splitActions[idx])

			// Only store the features we selected for this
			// tree to avoid excess memory usage.
			// This is likely to be necessary on games with
			// long rollouts.
			var selected attrMap
			switch data := inputVec.Data().(type) {
			case []float32:
				for _, j := range featurePerm {
					selected = append(selected, float64(data[j]))
				}
			case []float64:
				for _, j := range featurePerm {
					selected = append(selected, data[j])
				}
			default:
				panic("unsupported numeric type")
			}

			samples = append(samples, &sample{
				base:    selected,
				mapping: mapping,
				class:   action,
			})
		}
	}

	if len(samples) == 0 {
		return nil
	}

	if t.BuildTree != nil {
		return t.BuildTree(samples, features)
	} else {
		return idtrees.ID3(samples, features, 0)
	}
}

func (t *Trainer) bestRolloutIndices(r *anyrl.RolloutSet) []int {
	rewards := r.Rewards.Totals()
	indices := make([]int, len(rewards))
	for i := range indices {
		indices[i] = i
	}
	essentials.VoodooSort(rewards, func(i, j int) bool {
		return rewards[i] > rewards[j]
	}, indices)
	numSelect := int(math.Ceil(float64(len(indices)) * t.rolloutFrac()))
	return indices[:numSelect]
}

func (t *Trainer) rolloutFrac() float64 {
	if t.RolloutFrac == 0 {
		return 0.5
	}
	return t.RolloutFrac
}

func (t *Trainer) useFeatures() int {
	if t.UseFeatures == 0 {
		return t.NumFeatures
	}
	return t.UseFeatures
}

func (t *Trainer) useSteps(total int) float64 {
	if t.UseSteps != nil {
		return t.UseSteps(total)
	}
	return 1
}

type sample struct {
	base    idtrees.AttrMap
	mapping map[idtrees.Attr]idtrees.Attr
	class   int
}

func (s *sample) Attr(k idtrees.Attr) idtrees.Val {
	return s.base.Attr(s.mapping[k])
}

func (s *sample) Class() idtrees.Class {
	return s.class
}

func splitBatch(b *anyseq.Batch) []anyvec.Vector {
	vec := b.Packed
	idx := 0
	res := make([]anyvec.Vector, len(b.Present))
	subLen := vec.Len() / b.NumPresent()
	for i, pres := range b.Present {
		if pres {
			res[i] = b.Packed.Slice(idx, idx+subLen)
			idx += subLen
		}
	}
	return res
}
