package treeagent

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// A Roller produces anyrl.RolloutSets by running a policy
// in one or more environments.
type Roller struct {
	// Policy is used to sample actions.
	Policy *Tree

	// Creator is the anyvec.Creator behind vectors in the
	// environment(s) that this Roller will be using.
	Creator anyvec.Creator

	// NumActions is the total number of actions.
	NumActions int

	// These functions are called to produce tapes when
	// building a RolloutSet.
	//
	// You can set these in order to use special storage
	// techniques (e.g. compression or on-disk storage).
	//
	// For nil fields, lazyseq.ReferenceTape is used.
	MakeInputTape    anyrl.TapeMaker
	MakeActionTape   anyrl.TapeMaker
	MakeAgentOutTape anyrl.TapeMaker
}

// Rollout produces a rollout per environment.
func (r *Roller) Rollout(envs ...anyrl.Env) (*anyrl.RolloutSet, error) {
	res, err := r.rnnRoller().Rollout(envs...)
	return res, essentials.AddCtx("rollout tree", err)
}

func (r *Roller) rnnRoller() *anyrl.RNNRoller {
	return &anyrl.RNNRoller{
		Block: &anyrnn.FuncBlock{
			Func: func(in, state anydiff.Res, batch int) (out,
				newState anydiff.Res) {
				features := vecToFloats(in.Output())
				numFeatures := len(features) / batch

				var outProbs []float64
				for i := 0; i < batch; i++ {
					subFeatures := features[i*numFeatures : (i+1)*numFeatures]
					classDist := r.Policy.Find(subFeatures)
					outProbs = append(outProbs, classDist...)
				}

				vecData := r.Creator.MakeNumericList(outProbs)
				out = anydiff.NewConst(r.Creator.MakeVectorData(vecData))
				newState = state
				return
			},
			MakeStart: func(n int) anydiff.Res {
				return anydiff.NewConst(r.Creator.MakeVector(0))
			},
		},
		ActionSpace:      anyrl.Softmax{},
		MakeInputTape:    r.MakeInputTape,
		MakeActionTape:   r.MakeActionTape,
		MakeAgentOutTape: r.MakeAgentOutTape,
	}
}
