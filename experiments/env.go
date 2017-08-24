package experiments

import (
	"io"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
)

// ActionSpace is used to parameterize actions for an
// environment.
type ActionSpace interface {
	anyrl.LogProber
	anyrl.Sampler
	anyrl.Entropyer
}

// Env is an environment with a Close() method for
// releasing the environment's resources.
type Env interface {
	io.Closer
	anyrl.Env
}

// CloseEnvs closes every environment in the list.
func CloseEnvs(envs []Env) {
	for _, e := range envs {
		e.Close()
	}
}

// historyEnv keeps track of the previous observation and
// concatenates it with the current observation.
type historyEnv struct {
	Env

	lastObs anyvec.Vector
}

func (h *historyEnv) Reset() (anyvec.Vector, error) {
	obs, err := h.Env.Reset()
	h.lastObs = obs
	return h.nextObs(obs), err
}

func (h *historyEnv) Step(action anyvec.Vector) (anyvec.Vector, float64, bool, error) {
	obs, rew, done, err := h.Env.Step(action)
	return h.nextObs(obs), rew, done, err
}

func (h *historyEnv) nextObs(obs anyvec.Vector) anyvec.Vector {
	if obs == nil {
		return nil
	}
	res := obs.Creator().Concat(h.lastObs, obs)
	h.lastObs = obs
	return res
}
