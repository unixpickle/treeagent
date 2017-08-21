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

	// FeatureFrac is the fraction of features to try for
	// each branching node.
	//
	// If 0, all features are tried.
	FeatureFrac float64
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
// suitable for the Train method.
//
// The Advantage of the resulting samples is a low-bias
// estimator of the advantage function.
func (j *Judger) TrainingSamples(r *anyrl.RolloutSet) <-chan Sample {
	// The derivative of 0.5*(actual - predicted)^2 is just
	// the difference, which can be computed with GAE if
	// we set lambda to 1.
	j1 := *j
	j1.Lambda = 1
	differences := j1.JudgeActions(r)
	return RolloutSamples(r, differences)
}

// Train generates a tree to improve the value function.
//
// The advantages in the samples should come from
// TrainingSamples.
func (j *Judger) Train(data []Sample, maxDepth int) *Tree {
	var gradSamples []*gradientSample
	for _, sample := range data {
		gradSamples = append(gradSamples, &gradientSample{
			Sample:   sample,
			Gradient: []float64{sample.Advantage()},
		})
	}
	builder := &Builder{
		Algorithm:   MSEAlgorithm,
		FeatureFrac: j.FeatureFrac,
	}
	return builder.buildTree(gradSamples, maxDepth)
}

// OptimalWeight returns the optimal weight for the tree
// to improve the value function.
//
// The advantages in the samples should come from
// TrainingSamples.
func (j *Judger) OptimalWeight(data []Sample, t *Tree) float64 {
	var numerator float64
	var denominator float64
	for _, sample := range data {
		out := t.FindFeatureSource(sample)[0]
		denominator += out * out
		numerator += out * sample.Advantage()
	}
	return numerator / denominator
}
