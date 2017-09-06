package treeagent

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/essentials"
)

// A Roller produces anyrl.RolloutSets by running a policy
// in one or more environments.
type Roller struct {
	// Policy is used to sample actions.
	Policy *Forest

	// ActionSpace produces actions from parameters.
	ActionSpace anyrl.Sampler

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

// Creator returns the creator that is used to create
// vectors.
func (r *Roller) Creator() anyvec.Creator {
	return anyvec64.DefaultCreator{}
}

func (r *Roller) rnnRoller() *anyrl.RNNRoller {
	return &anyrl.RNNRoller{
		Creator: r.Creator(),
		Block: &anyrnn.FuncBlock{
			Func: func(in, state anydiff.Res, batch int) (out,
				newState anydiff.Res) {
				out = anydiff.NewConst(r.Policy.applyBatch(in.Output(), batch))
				newState = state
				return
			},
			MakeStart: func(n int) anydiff.Res {
				return anydiff.NewConst(r.Creator().MakeVector(0))
			},
		},
		ActionSpace:      r.ActionSpace,
		MakeInputTape:    r.MakeInputTape,
		MakeActionTape:   r.MakeActionTape,
		MakeAgentOutTape: r.MakeAgentOutTape,
	}
}
