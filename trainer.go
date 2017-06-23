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
}

// Train trains a random forest on the rollouts.
func (t *Trainer) Train(r *anyrl.RolloutSet) idtrees.Forest {
	rewards := r.Rewards.Totals()
	indices := make([]int, len(rewards))
	for i := range indices {
		indices[i] = i
	}
	essentials.VoodooSort(rewards, func(i, j int) bool {
		return rewards[i] > rewards[j]
	}, indices)
	numSelect := int(math.Ceil(float64(len(indices)) * t.RolloutFrac))
	indices = indices[:numSelect]

	var res idtrees.Forest
	for i := 0; i < t.NumTrees; i++ {
		tree := t.buildTree(r, indices)
		if tree != nil {
			res = append(res, tree)
		}
	}
	return res
}

func (t *Trainer) buildTree(r *anyrl.RolloutSet, indices []int) *idtrees.Tree {
	numFeatures := int(math.Ceil(math.Sqrt(float64(t.NumFeatures))))
	features := []idtrees.Attr{}
	mapping := map[idtrees.Attr]idtrees.Attr{}
	featurePerm := rand.Perm(t.NumFeatures)[:numFeatures]
	for i, j := range featurePerm {
		mapping[j] = i
		features = append(features, j)
	}

	totalSamples := 0
	for _, idx := range indices {
		totalSamples += len(r.Rewards[idx])
	}
	sampleProb := 1 / float64(math.Sqrt(float64(totalSamples)))

	var samples []idtrees.Sample
	actions := r.Actions.ReadTape(0, -1)
	for input := range r.Inputs.ReadTape(0, -1) {
		split := splitBatch(input)
		splitActions := splitBatch(<-actions)
		for _, idx := range indices {
			if !input.Present[idx] || rand.Float64() > sampleProb {
				continue
			}
			vec := split[idx]
			var selected attrMap
			switch data := vec.Data().(type) {
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
			action := anyvec.MaxIndex(splitActions[idx])
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

	return idtrees.ID3(samples, features, 0)
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
	subLen := vec.Len() / len(res)
	for i, pres := range b.Present {
		if pres {
			res[i] = b.Packed.Slice(idx, idx+subLen)
			idx += subLen
		}
	}
	return res
}
