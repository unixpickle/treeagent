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
// It returns a tree approximation of the gradient, the
// mean objective, and the mean regulizer (or 0).
func (p *PPO) Step(s []Sample, f *Forest) (step *Tree, obj, reg anyvec.Numeric) {
	objective, grad := computeObjective(s, f, p.objective)
	grad = p.Builder.maskGradients(grad)
	objParts := vecToFloats(objective)
	scaler := 1 / float64(len(s))
	return p.Builder.buildTree(grad, grad, p.Builder.MaxDepth),
		scaler * objParts[0], scaler * objParts[1]
}

// objective computes the mean PPO objective concatenated
// with the regularization term (or 0).
func (p *PPO) objective(params, oldParams, acts, advs anydiff.Res, n int) anydiff.Res {
	c := params.Output().Creator()

	oldProbs := p.Builder.ActionSpace.LogProb(oldParams, acts.Output(), n)
	newProbs := p.Builder.ActionSpace.LogProb(params, acts.Output(), n)
	ratios := anydiff.Exp(anydiff.Sub(newProbs, oldProbs))

	epsilon := p.Epsilon
	if epsilon == 0 {
		epsilon = anypg.DefaultPPOEpsilon
	}
	obj := anydiff.Sum(anypg.PPOObjective(c.MakeNumeric(epsilon), ratios, advs))

	if p.Builder.Regularizer != nil {
		reg := p.Builder.Regularizer.Regularize(params, n)
		obj = anydiff.Concat(obj, reg)
	} else {
		obj = anydiff.Concat(obj, anydiff.NewConst(c.MakeVector(1)))
	}

	return obj
}
