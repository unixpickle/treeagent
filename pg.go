package treeagent

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
)

// A PG implements policy gradient optimization.
type PG struct {
	Builder Builder

	// ActionSpace is used to determine the probability of
	// actions given the action parameters.
	ActionSpace anyrl.LogProber

	// Regularizer, if non-nil, is used to regularize the
	// action distributions of the policy.
	Regularizer anypg.Regularizer
}

// Build approximates the policy gradient with a tree.
// It returns the tree, the surrogate objective, and the
// regularization term.
func (p *PG) Build(data []Sample) (step *Tree, obj, reg anyvec.Numeric) {
	return p.Builder.buildWithTerms(computeObjective(data, nil, p.Objective))
}

// Objective implements the policy gradient objective
// function with (optional) regularization.
// The first dimension of the output is for policy
// gradients.
// The second dimension is for regularization.
func (p *PG) Objective(params, old, acts, advs anydiff.Res, n int) anydiff.Res {
	probs := p.ActionSpace.LogProb(params, acts.Output(), n)
	obj := anydiff.Sum(anydiff.Mul(probs, advs))

	if p.Regularizer != nil {
		reg := p.Regularizer.Regularize(params, n)
		obj = anydiff.Concat(obj, anydiff.Sum(reg))
	} else {
		c := obj.Output().Creator()
		obj = anydiff.Concat(obj, anydiff.NewConst(c.MakeVector(1)))
	}

	return obj
}
