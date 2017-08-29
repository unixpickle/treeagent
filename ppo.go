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
	// PG is used to configure how individual trees are
	// built during PPO.
	PG PG

	// Epsilon is the amount by which the probability ratio
	// should change.
	//
	// If 0, anypg.DefaultPPOEpsilon is used.
	Epsilon float64
}

// Build performs a single step of PPO on the samples.
//
// It returns a tree approximation of the gradient, the
// mean objective, and the mean regulizer (or 0).
func (p *PPO) Build(s []Sample, f *Forest) (step *Tree, obj, reg anyvec.Numeric) {
	return p.PG.Builder.buildWithTerms(computeObjective(s, f, p.Objective))
}

// WeightGradient returns the gradient with respect to the
// tree weights.
//
// The gradient and objective are means over all samples.
// Thus, the scale of the result does not depend on the
// number of samples.
func (p *PPO) WeightGradient(s []Sample, f *Forest) (grad []float64, obj,
	reg anyvec.Numeric) {
	grad, rawObj := weightGradient(s, f, p.Objective)
	obj, reg = splitUpTerms(rawObj, len(s))
	return
}

// Objective computes the  PPO objective concatenated with
// the regularization (or 0 if no regularization is used).
func (p *PPO) Objective(params, oldParams, acts, advs anydiff.Res, n int) anydiff.Res {
	c := params.Output().Creator()

	oldProbs := p.PG.ActionSpace.LogProb(oldParams, acts.Output(), n)
	newProbs := p.PG.ActionSpace.LogProb(params, acts.Output(), n)
	ratios := anydiff.Exp(anydiff.Sub(newProbs, oldProbs))

	epsilon := p.Epsilon
	if epsilon == 0 {
		epsilon = anypg.DefaultPPOEpsilon
	}
	obj := anydiff.Sum(anypg.PPOObjective(c.MakeNumeric(epsilon), ratios, advs))

	if p.PG.Regularizer != nil {
		reg := p.PG.Regularizer.Regularize(params, n)
		obj = anydiff.Concat(obj, anydiff.Sum(reg))
	} else {
		obj = anydiff.Concat(obj, anydiff.NewConst(c.MakeVector(1)))
	}

	return obj
}
