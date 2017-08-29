package treeagent

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/lazyseq"
)

// A Judger trains and uses a value-function approximator
// to compute action advantages.
// It can be used to reduce variance during training.
type Judger struct {
	// ValueFunc takes input features for a state and
	// predicts the mean reward following that state.
	ValueFunc *Forest

	// Discount is the reward discount factor.
	Discount float64

	// Lambda is the GAE parameter.
	// A lambda of 0 is high-bias and low-variance.
	// A lambda of 1 is low-bias and high-variance.
	//
	// For more on GAE, see:
	// https://arxiv.org/abs/1506.02438.
	Lambda float64

	// These options are the same as those in Builder.
	MaxDepth    int
	FeatureFrac float64
	MinLeaf     int
}

// JudgeActions produces advantage estimations.
func (j *Judger) JudgeActions(r *anyrl.RolloutSet) anyrl.Rewards {
	judger := &anypg.GAEJudger{
		ValueFunc: func(seq lazyseq.Rereader) <-chan *anyseq.Batch {
			return lazyseq.Map(seq, func(v anydiff.Res, n int) anydiff.Res {
				return anydiff.NewConst(j.ValueFunc.applyBatch(v.Output(), n))
			}).Forward()
		},
		Discount: j.Discount,
		Lambda:   j.Lambda,
	}
	return judger.JudgeActions(r)
}

// TrainingSamples produces a stream of Samples which are
// suitable for the Train and OptimalWeight methods.
//
// One set of training samples can be re-used for multiple
// training iterations.
func (j *Judger) TrainingSamples(r *anyrl.RolloutSet) <-chan Sample {
	judger := &anypg.QJudger{Discount: j.Discount}
	return RolloutSamples(r, judger.JudgeActions(r))
}

// Train generates a tree to improve the value function
// and returns the loss that the tree aims to improve.
//
// The advantages in the samples should come from
// TrainingSamples.
func (j *Judger) Train(data []Sample) (*Tree, float64) {
	var gradSamples []*gradientSample
	var loss float64
	outs := j.ValueFunc.applySamples(data)
	for i, sample := range data {
		grad := sample.Advantage() - outs[i][0]
		gradSamples = append(gradSamples, &gradientSample{
			Sample:   sample,
			Gradient: []float64{grad},
		})
		loss += grad * grad
	}
	builder := Builder{
		Algorithm:   MSEAlgorithm,
		MaxDepth:    j.MaxDepth,
		FeatureFrac: j.FeatureFrac,
		MinLeaf:     j.MinLeaf,
	}
	mse := loss / float64(len(data))
	return builder.build(gradSamples), mse
}

// OptimalWeight returns the optimal weight for the tree
// to improve the value function.
//
// The advantages in the samples should come from
// TrainingSamples.
func (j *Judger) OptimalWeight(data []Sample, t *Tree) float64 {
	var numerator float64
	var denominator float64
	outs := j.ValueFunc.applySamples(data)
	for i, sample := range data {
		out := t.FindFeatureSource(sample)[0]
		approximation := outs[i][0]
		denominator += out * out
		numerator += out * (sample.Advantage() - approximation)
	}
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}
