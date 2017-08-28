package treeagent

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// objectiveFunc is an optimization objective function.
// It takes in a batch of parameters, actions, and
// advantages and produces an objective value to maximize.
//
// The n argument specifies how many samples are
// represented by the batch.
//
// The params argument is the current value of the action
// parameters, which are to be learned.
// The oldParams argument is the constant ActionParams for
// each sample.
//
// The objective may be more than one-dimensional.
// In this case, the sum is optimized.
type objectiveFunc func(params, oldParams, acts, advs anydiff.Res, n int) anydiff.Res

// computeObjective computes the objective function and
// its gradient with respect to the action parameters.
//
// If f is non-nil, it is applied to the samples in
// order to compute the params vector.
// Otherwise, the ActionParams of each Sample are used.
// The f argument is only necessary in offline-policy
// algorithms or where the samples are re-used for
// multiple steps.
func computeObjective(s []Sample, f *Forest, o objectiveFunc) (anyvec.Vector,
	[]*gradientSample) {
	oldParams := make([]anyvec.Vector, len(s))
	actions := make([]anyvec.Vector, len(s))
	advs := make([]float64, len(s))
	for i, sample := range s {
		advs[i] = sample.Advantage()
		oldParams[i] = sample.ActionParams()
		actions[i] = sample.Action()
	}
	c := actions[0].Creator()
	oldParamRes := anydiff.NewConst(c.Concat(oldParams...))
	actRes := anydiff.NewConst(c.Concat(actions...))
	advRes := anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(advs)))

	var newParamRes *anydiff.Var
	if f != nil {
		var joined []float64
		for _, out := range f.applySamples(s) {
			joined = append(joined, out...)
		}
		newParamRes = anydiff.NewVar(c.MakeVectorData(c.MakeNumericList(joined)))
	} else {
		newParamRes = anydiff.NewVar(oldParamRes.Output())
	}

	objective := o(newParamRes, oldParamRes, actRes, advRes, len(s))
	grad := splitSampleGrads(s, newParamRes, anydiff.Sum(objective))
	return objective.Output(), grad
}

// gradientSample is a Sample paired with the gradient of
// some objective with respect to the sample's parameter
// vector.
type gradientSample struct {
	Sample
	Gradient smallVec
}

// splitSampleGrads takes the gradient of obj with respect
// to params and splits it up amongst the samples.
func splitSampleGrads(samples []Sample, params *anydiff.Var,
	obj anydiff.Res) []*gradientSample {
	grad := anydiff.NewGrad(params)
	obj.Propagate(anyvec.Ones(params.Output().Creator(), 1), grad)
	gradVec := grad[params]
	gradSize := gradVec.Len() / len(samples)

	res := make([]*gradientSample, len(samples))
	nativeGrad := vecToFloats(gradVec)
	for i, s := range samples {
		res[i] = &gradientSample{
			Sample:   s,
			Gradient: nativeGrad[i*gradSize : (i+1)*gradSize],
		}
	}
	return res
}

// sumGradients computes the sum of the sample's
// gradients.
func sumGradients(samples []*gradientSample) smallVec {
	sum := samples[0].Gradient.Copy()
	for _, sample := range samples[1:] {
		sum.Add(sample.Gradient)
	}
	return sum
}
