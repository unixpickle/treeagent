package experiments

import (
	"errors"
	"io"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
)

// ActionSpace is used to parameterize actions for an
// environment.
type ActionSpace interface {
	anyrl.LogProber
	anyrl.Sampler
	anyrl.Entropyer
}

// EnvInfo stores information about an environment.
type EnvInfo struct {
	// Name of the environment.
	Name string

	// Information about actions and their parameters.
	ActionSpace ActionSpace
	ParamSize   int

	// Screen information (may be subsampled).
	// Not applicable for some environments.
	Width  int
	Height int

	// Number of features (e.g. number of pixels).
	NumFeatures int

	// The collection to which the environment belongs.
	Muniverse bool
	Atari     bool
}

// LookupEnvInfo finds information about an environment.
func LookupEnvInfo(name string) (*EnvInfo, error) {
	spec := muniverse.SpecForName(name)
	if spec != nil {
		w, h := muniverseDownsampledSize(spec.Width, spec.Height)
		res := &EnvInfo{
			Name:        name,
			ActionSpace: anyrl.Softmax{},
			ParamSize:   len(spec.KeyWhitelist) + 1,
			Width:       w,
			Height:      h,
			NumFeatures: w * h,
			Muniverse:   true,
		}
		if res.ParamSize == 1 {
			// Support tapping games with no keyboard.
			res.ActionSpace = &anyrl.Bernoulli{OneHot: true}
		}
		return res, nil
	}

	if numActions, ok := atariActionSizes[name]; ok {
		return &EnvInfo{
			Name:        name,
			ActionSpace: anyrl.Softmax{},
			ParamSize:   numActions,
			Width:       atariWidth,
			Height:      atariHeight,
			NumFeatures: atariObsSize(name),
			Atari:       true,
		}, nil
	}

	return nil, errors.New("lookup game environment: \"" + name + "\" not found")
}

// Env is an environment with a Close() method for
// releasing the environment's resources.
type Env interface {
	io.Closer
	anyrl.Env
}

// MakeEnvs creates n instances of an environment.
func MakeEnvs(c anyvec.Creator, e *EnvFlags, n int) (envs []Env, err error) {
	defer essentials.AddCtxTo("make games ("+e.Name+")", &err)
	info, err := LookupEnvInfo(e.Name)
	if err != nil {
		return nil, err
	}
	if info.Muniverse {
		return newMuniverseEnvs(c, e, n)
	} else if info.Atari {
		return newAtariEnvs(c, e, n)
	} else {
		return nil, errors.New("unknown game source")
	}
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
	res := obs.Creator().Concat(obs, h.lastObs)
	h.lastObs = obs
	return res
}
