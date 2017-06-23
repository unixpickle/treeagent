package treeagent

import (
	"math"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/weakai/idtrees"
)

// A Roller runs a Policy on a set of environments.
type Roller struct {
	Policy  *Policy
	Creator anyvec.Creator

	// Used to produce tapes for a RolloutSet.
	//
	// For nil fields, lazyseq.ReferenceTape is used.
	MakeInputTape    anyrl.TapeMaker
	MakeActionTape   anyrl.TapeMaker
	MakeAgentOutTape anyrl.TapeMaker
}

// Rollout produces a rollout per environment.
func (r *Roller) Rollout(envs ...anyrl.Env) (*anyrl.RolloutSet, error) {
	res, err := r.rnnRoller().Rollout(envs...)
	return res, essentials.AddCtx("rollout policy", err)
}

func (r *Roller) rnnRoller() *anyrl.RNNRoller {
	return &anyrl.RNNRoller{
		Block: &anyrnn.FuncBlock{
			Func: func(in, state anydiff.Res, batch int) (out,
				newState anydiff.Res) {
				var attrs attrMap
				switch data := in.Output().Data().(type) {
				case []float64:
					attrs = data
				case []float32:
					for _, x := range data {
						attrs = append(attrs, float64(x))
					}
				default:
					panic("unsupported numeric type")
				}
				dist := r.Policy.Classify(attrs)
				vec := make([]float64, r.Policy.NumActions)
				for i := range vec {
					if val, ok := dist[i]; ok {
						vec[i] = math.Log(val)
					}
				}
				vecData := r.Creator.MakeNumericList(vec)
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

type attrMap []float64

func (a attrMap) Attr(idx idtrees.Attr) idtrees.Val {
	return a[idx.(int)]
}
