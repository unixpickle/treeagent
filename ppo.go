package treeagent

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
)

// PPO implements a tree-based variant of Proximal Policy
// Optimization.
//
// See the PPO paper: https://arxiv.org/abs/1707.06347.
type PPO struct {
	// Builder is used to configure how individual trees
	// are built during PPO.
	Builder *Builder

	// Epsilon is the amount by which the probability ratio
	// should change.
	//
	// If 0, anypg.DefaultPPOEpsilon is used.
	Epsilon float64
}

// Step performs a single step of PPO on the samples.
//
// It returns a tree approximation of the gradient.
// It also returns the current value of the surrogate
// objective, which the tree is designed to maximize.
func (p *PPO) Step(data []Sample, forest *Forest) (step *Tree, obj anyvec.Numeric) {
	params := p.forestParams(data, forest)
	objective := p.objective(data, params)
	gradSamples := splitSampleGrads(data, params, objective)
	c := objective.Output().Creator()
	objMean := c.NumOps().Div(anyvec.Sum(objective.Output()),
		c.MakeNumeric(float64(len(data))))
	gradSamples = p.Builder.maskGradients(gradSamples)
	return p.Builder.buildTree(gradSamples, p.Builder.MaxDepth), objMean
}

// forestParams applies the forest to each sample and
// returns a variable containing every output vector.
func (p *PPO) forestParams(samples []Sample, forest *Forest) *anydiff.Var {
	c := samples[0].Action().Creator()
	var outVecs []float64
	for _, sample := range samples {
		forestOut := forest.ApplyFeatureSource(sample)
		outVecs = append(outVecs, forestOut...)
	}
	return anydiff.NewVar(c.MakeVectorData(c.MakeNumericList(outVecs)))
}

// objective computes the mean PPO objective.
func (p *PPO) objective(samples []Sample, params anydiff.Res) anydiff.Res {
	c := samples[0].Action().Creator()

	oldParams := make([]anyvec.Vector, len(samples))
	actions := make([]anyvec.Vector, len(samples))
	advs := make([]float64, len(samples))
	for i, sample := range samples {
		advs[i] = sample.Advantage()
		oldParams[i] = sample.ActionParams()
		actions[i] = sample.Action()
	}
	oldParamRes := anydiff.NewConst(c.Concat(oldParams...))
	actionsVec := c.Concat(actions...)
	advRes := anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(advs)))

	oldProbs := p.Builder.ActionSpace.LogProb(oldParamRes, actionsVec, len(samples))
	newProbs := p.Builder.ActionSpace.LogProb(params, actionsVec, len(samples))
	ratios := anydiff.Exp(anydiff.Sub(newProbs, oldProbs))

	epsilon := p.Epsilon
	if epsilon == 0 {
		epsilon = anypg.DefaultPPOEpsilon
	}
	obj := anypg.PPOObjective(c.MakeNumeric(epsilon), ratios, advRes)

	if p.Builder.Regularizer != nil {
		obj = anydiff.Add(obj, p.Builder.Regularizer.Regularize(params, len(samples)))
	}

	return anydiff.Sum(obj)
}
