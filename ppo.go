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
	var gradSamples []*gradientSample
	var objSum anyvec.Numeric
	for _, sample := range data {
		obj, grad := p.objective(sample, forest)
		gradSamples = append(gradSamples, &gradientSample{
			Sample:   sample,
			Gradient: grad,
		})
		if objSum == nil {
			objSum = obj
		} else {
			objSum = grad.Creator().NumOps().Add(objSum, obj)
		}
	}
	if objSum != nil {
		c := gradSamples[0].Action().Creator()
		objSum = c.NumOps().Div(objSum, c.MakeNumeric(float64(len(data))))
	}
	return p.Builder.buildTree(gradSamples, p.Builder.MaxDepth), objSum
}

func (p *PPO) objective(sample Sample, forest *Forest) (obj anyvec.Numeric,
	grad anyvec.Vector) {
	c := sample.Action().Creator()

	forestOut := forest.ApplyFeatureSource(sample)
	outVar := anydiff.NewVar(c.MakeVectorData(c.MakeNumericList(forestOut)))

	oldProb := p.Builder.ActionSpace.LogProb(
		anydiff.NewConst(sample.ActionParams()),
		sample.Action(),
		1,
	)
	newProb := p.Builder.ActionSpace.LogProb(outVar, sample.Action(), 1)
	ratio := anydiff.Exp(anydiff.Sub(newProb, oldProb))
	rawObj := anydiff.Scale(ratio, c.MakeNumeric(sample.Advantage()))

	if p.shouldClip(rawObj, sample.Advantage()) {
		best := p.bestValue(sample.Advantage())
		rawObj = anydiff.NewConst(c.MakeVectorData(c.MakeNumericList([]float64{best})))
	}

	obj = anyvec.Sum(rawObj.Output())

	if p.Builder.Regularizer != nil {
		rawObj = anydiff.Add(rawObj, p.Builder.Regularizer.Regularize(outVar, 1))
	}

	g := anydiff.NewGrad(outVar)
	rawObj.Propagate(anyvec.Ones(c, 1), g)
	return obj, g[outVar]
}

func (p *PPO) shouldClip(rawObj anydiff.Res, adv float64) bool {
	c := rawObj.Output().Creator()
	ops := c.NumOps()
	best := c.MakeNumeric(p.bestValue(adv))
	actual := anyvec.Sum(rawObj.Output())
	return ops.Greater(actual, best)
}

func (p *PPO) bestValue(adv float64) float64 {
	epsilon := p.Epsilon
	if epsilon == 0 {
		epsilon = anypg.DefaultPPOEpsilon
	}
	if adv < 0 {
		return adv * (1 - epsilon)
	} else {
		return adv * (1 + epsilon)
	}
}
