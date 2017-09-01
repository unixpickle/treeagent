package treeagent

import (
	"runtime"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// ObjectiveFunc is an optimization objective function.
// It takes in a batch of parameters, actions, and
// advantages and produces an objective value to maximize.
//
// The params argument is the current value of the action
// parameters, which are to be updated.
//
// The oldParams argument contains the action parameters
// which were originally output and resulted in the
// sampled actions.
//
// The acts argument contains the sampled action vectors.
//
// The advs argument stores, for each sample, an estimated
// advantage value.
//
// The n argument specifies how many samples are
// represented by the batch.
//
// Generally, the return value should be a single number.
// However, it may be multi-dimensional so as to separate
// different terms of the objective (e.g. the regulizer
// and the policy gradient).
// In this case, the sum of the vector components is used
// as the final objective value to maximize.
//
// The terms in the result should not be normalized by
// dividing by n.
// Rather, the objective should represent a sum.
type ObjectiveFunc func(params, oldParams, acts, advs anydiff.Res, n int) anydiff.Res

// Improved checks if a policy makes an improvement over
// the policy that originally produced the samples.
func Improved(s []Sample, f *Forest, o ObjectiveFunc) bool {
	newParams, oldParams, acts, advs := objectiveArguments(s, f, o)
	newObj := anyvec.Sum(o(newParams, oldParams, acts, advs, len(s)).Output())
	oldObj := anyvec.Sum(o(oldParams, oldParams, acts, advs, len(s)).Output())
	return acts.Output().Creator().NumOps().Greater(newObj, oldObj)
}

// weightGradient computes the gradient of an objective
// with respect to the weights in a forest.
// It returns the value of the objective function and the
// gradient with respect to the weights.
//
// The gradient is divided by the total number of samples,
// ensuring that it is invariant to the sample count.
func weightGradient(s []Sample, f *Forest, o ObjectiveFunc) (grad []float64,
	obj anyvec.Vector) {
	obj, gradSamples := computeObjective(s, f, o)
	grad = make([]float64, len(f.Trees))

	indices := make(chan int, len(grad))
	for i := range grad {
		indices <- i
	}
	close(indices)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range indices {
				tree := f.Trees[index]
				grad[index] = treeWeightGradient(gradSamples, tree)
			}
		}()
	}

	wg.Wait()
	return
}

func treeWeightGradient(g []*gradientSample, t *Tree) float64 {
	var sum float64
	for _, sample := range g {
		sum += sample.Gradient.Dot(smallVec(t.FindFeatureSource(sample)))
	}
	return sum / float64(len(g))
}

// computeObjective computes the objective function and
// its gradient with respect to the action parameters.
//
// If f is non-nil, it is applied to the samples in
// order to compute the params vector.
// Otherwise, the ActionParams of each Sample are used.
// The f argument is only necessary in offline-policy
// algorithms or where the samples are re-used for
// multiple steps.
func computeObjective(s []Sample, f *Forest, o ObjectiveFunc) (anyvec.Vector,
	[]*gradientSample) {
	newParams, oldParams, acts, advs := objectiveArguments(s, f, o)
	objective := o(newParams, oldParams, acts, advs, len(s))
	grad := splitSampleGrads(s, newParams, anydiff.Sum(objective))
	return objective.Output(), grad
}

// objectiveArguments produces the arguments for an
// objective function.
func objectiveArguments(s []Sample, f *Forest, o ObjectiveFunc) (*anydiff.Var,
	*anydiff.Const, *anydiff.Const, *anydiff.Const) {
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
	return newParamRes, oldParamRes, actRes, advRes
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

// splitUpTerms splits up the objective vector into its
// two components: surrogate loss and regularization.
// It divides both terms by n.
func splitUpTerms(objAndReg anyvec.Vector, n int) (obj, reg anyvec.Numeric) {
	objParts := vecToFloats(objAndReg)
	scaler := 1 / float64(n)
	return scaler * objParts[0], scaler * objParts[1]
}
