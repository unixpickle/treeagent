package experiments

import (
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/cuberl"
)

const (
	cubeRLNumActs = 18
	cubeRLNumObs  = 8*6 + 1
)

func cubeRLInfo() *EnvInfo {
	return &EnvInfo{
		Name:        "Cube",
		ActionSpace: anyrl.Softmax{},
		ParamSize:   cuberl.NumActions,
		NumFeatures: cubeRLNumObs,
		CubeRL:      true,
	}
}

type cuberlEnv struct {
	anyrl.Env
}

func newCubeRLEnvs(c anyvec.Creator, e *EnvFlags, n int) ([]Env, error) {
	var res []Env
	for i := 0; i < n; i++ {
		// TODO: flags for some of these parameters.
		res = append(res, &cuberlEnv{
			Env: &cuberl.Env{
				Objective: cuberl.FullCube,
				EpLen:     20,
				FullState: true,
			},
		})
	}
	return res, nil
}

func (c *cuberlEnv) Close() error {
	return nil
}
