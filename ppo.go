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
// It returns a tree which should be added to forest.
func (p *PPO) Step(data []Sample, forest *Forest) *Tree {
	var gradSamples []*gradientSample
	for _, sample := range data {
		gradSamples = append(gradSamples, &gradientSample{
			Sample:   sample,
			Gradient: p.gradient(sample, forest),
		})
	}
	return p.Builder.buildTree(gradSamples, p.Builder.MaxDepth)
}

func (p *PPO) gradient(sample Sample, forest *Forest) anyvec.Vector {
	c := sample.Action().Creator()

	// The vector in sample.ActionParams() may be out of
	// date if the forest was trained more.
	features := make([]float64, p.Builder.NumFeatures)
	for i := range features {
		features[i] = sample.Feature(i)
	}
	forestOut := forest.Apply(features)
	outVar := anydiff.NewVar(c.MakeVectorData(c.MakeNumericList(forestOut)))

	oldProb := p.Builder.ActionSpace.LogProb(
		anydiff.NewConst(sample.ActionParams()),
		sample.Action(),
		1,
	)
	newProb := p.Builder.ActionSpace.LogProb(outVar, sample.Action(), 1)
	ratio := anydiff.Exp(anydiff.Sub(newProb, oldProb))

	if p.shouldClip(sample, ratio) {
		return c.MakeVector(sample.ActionParams().Len())
	}

	obj := anydiff.Scale(ratio, c.MakeNumeric(sample.Advantage()))
	grad := anydiff.NewGrad(outVar)
	obj.Propagate(anyvec.Ones(c, 1), grad)
	return grad[outVar]
}

func (p *PPO) shouldClip(sample Sample, ratio anydiff.Res) bool {
	epsilon := p.Epsilon
	if epsilon == 0 {
		epsilon = anypg.DefaultPPOEpsilon
	}

	c := ratio.Output().Creator()
	ops := c.NumOps()
	ratScalar := anyvec.Sum(ratio.Output())
	if sample.Advantage() > 0 {
		max := c.MakeNumeric(1 + epsilon)
		return ops.Greater(ratScalar, max)
	} else {
		min := c.MakeNumeric(0 - epsilon)
		return ops.Less(ratScalar, min)
	}
}
