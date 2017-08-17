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

// Train adds a tree to the value function based on the
// training samples.
//
// The advantages in the samples are used as the error
// gradient.
// If the samples originated from TrainingSamples, this is
// guaranteed to be a correct assumption.
func (j *Judger) Train(data []Sample, numFeatures int, maxDepth int, weight float64) {
	var gradSamples []*gradientSample
	for _, sample := range data {
		c := sample.Action().Creator()
		vec := c.MakeVectorData(c.MakeNumericList([]float64{sample.Advantage()}))
		gradSamples = append(gradSamples, &gradientSample{
			Sample:   sample,
			Gradient: vec,
		})
	}
	builder := &Builder{
		NumFeatures: numFeatures,
		Algorithm:   MSEAlgorithm,
	}
	tree := builder.buildTree(gradSamples, maxDepth)
	j.ValueFunc.Add(tree, weight)
}
