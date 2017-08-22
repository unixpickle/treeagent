package main

import (
	"fmt"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/treeagent"
)

// ExactGradient computes the sample-wise policy gradient.
func ExactGradient(samples []treeagent.Sample, space anyrl.LogProber) anyvec.Vector {
	var allParams []anyvec.Vector
	var allOuts []anyvec.Vector
	var allAdvs []float64
	for _, sample := range samples {
		allParams = append(allParams, sample.ActionParams())
		allOuts = append(allOuts, sample.Action())
		allAdvs = append(allAdvs, sample.Advantage())
	}
	c := samples[0].Action().Creator()
	params := anydiff.NewVar(c.Concat(allParams...))
	outs := c.Concat(allOuts...)
	advs := c.MakeVectorData(c.MakeNumericList(allAdvs))

	logProbs := space.LogProb(params, outs, len(samples))
	loss := anydiff.Sum(anydiff.Mul(logProbs, anydiff.NewConst(advs)))

	grad := anydiff.NewGrad(params)
	loss.Propagate(anyvec.Ones(c, 1), grad)
	return grad[params]
}

// GradAnalysis analyses the update induced by t in terms
// of the exact gradient.
func GradAnalysis(t *treeagent.Tree, samples []treeagent.Sample, grad anyvec.Vector) {
	c := grad.Creator()
	ops := c.NumOps()

	outGrad := outputGradient(t, samples)

	dot := outGrad.Dot(grad)
	cosDist := ops.Div(dot, ops.Mul(anyvec.Norm(outGrad), anyvec.Norm(grad)))
	fmt.Println("Gradient cosine distance:", cosDist)

	diff := outGrad.Copy()
	diff.Sub(grad)
	fmt.Println("Gradient L1 distance:", anyvec.AbsSum(diff))
	fmt.Println("Gradient L2 distance:", anyvec.Norm(diff))
}

func outputGradient(t *treeagent.Tree, samples []treeagent.Sample) anyvec.Vector {
	var allOuts []float64
	for _, sample := range samples {
		allOuts = append(allOuts, t.FindFeatureSource(sample)...)
	}
	c := samples[0].Action().Creator()
	return c.MakeVectorData(c.MakeNumericList(allOuts))
}
