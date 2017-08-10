package treeagent

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

// A Trainer uses a simplified form of policy gradients to
// update the action distributions for training samples.
// In other words, a Trainer pulls action distributions
// closer to the optimal distribution, provided there is
// enough sample data.
type Trainer struct {
	// StepSize controls how fast learning is done.
	// A smaller step size means smaller changes to
	// action distributions.
	StepSize float64

	// EntropyReg can be used to encourage exploration.
	// If it is non-zero, it should usually be a small
	// number like 0.01.
	EntropyReg float64

	// Judger is used to decide how good actions are.
	//
	// If nil, anypg.TotalJudger is used.
	Judger anypg.ActionJudger
}

// Targets derives a stream of Samples with updated action
// distributions based on the advantage values.
//
// The incoming channel of Samples should be in the order
// produced by RolloutSamples.
//
// The caller must read the entire channel to prevent a
// resource leak.
func (t *Trainer) Targets(r *anyrl.RolloutSet, samples <-chan Sample) <-chan Sample {
	res := make(chan Sample, 1)
	c := anyvec64.DefaultCreator{}
	go func() {
		defer close(res)
		advantages := flattenAdvantages(t.judger().JudgeActions(r))
		actions := selectedActions(r)
		for sample := range samples {
			advantage := advantages[0]
			advantages = advantages[1:]
			action := <-actions

			// Perform policy gradients on the action distribution,
			// treating it as a parameter vector for softmax.

			paramVec := anydiff.NewVar(c.MakeVectorData(sample.ActionDist))
			logProbs := anydiff.LogSoftmax(paramVec, -1)
			logProb := anydiff.Slice(logProbs, action, action+1)
			objective := anydiff.Scale(logProb, advantage)

			if t.EntropyReg != 0 {
				space := anyrl.Softmax{}
				regTerm := anydiff.Scale(space.Entropy(paramVec, 1), t.EntropyReg)
				objective = anydiff.Add(objective, regTerm)
			}

			grad := anydiff.NewGrad(paramVec)
			objective.Propagate(anyvec.Ones(c, 1), grad)
			grad.Scale(t.StepSize)
			grad.AddToVars()

			anyvec.LogSoftmax(paramVec.Vector, 0)
			res <- &updatedSample{
				Sample: sample,
				Dist:   paramVec.Vector.Data().([]float64),
			}
		}
	}()
	return res
}

func (t *Trainer) judger() anypg.ActionJudger {
	if t.Judger == nil {
		return &anypg.TotalJudger{Normalize: true}
	}
	return t.Judger
}

func flattenAdvantages(r anyrl.Rewards) []float64 {
	var res []float64
	for _, batch := range r {
		for _, x := range batch {
			res = append(res, x)
		}
	}
	return res
}

func selectedActions(r *anyrl.RolloutSet) <-chan int {
	res := make(chan int, 1)
	go func() {
		defer close(res)
		for timestep := range r.Actions.ReadTape(0, -1) {
			oneHots := timestep.Packed
			numActions := oneHots.Len() / timestep.NumPresent()
			for i := 0; i < numActions; i++ {
				oneHot := oneHots.Slice(i*numActions, (i+1)*numActions)
				res <- anyvec.MaxIndex(oneHot)
			}
		}
	}()
	return res
}

type updatedSample struct {
	Sample
	Dist ActionDist
}

func (u *updatedSample) ActionDist() ActionDist {
	return u.Dist
}
