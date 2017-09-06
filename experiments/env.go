package experiments

import (
	"compress/flate"
	"errors"
	"io"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/treeagent"
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

	// If true, features can be compressed as uint8
	// values to use less RAM.
	Uint8Features bool

	// The collection to which the environment belongs.
	Muniverse bool
	Atari     bool
	CubeRL    bool
	MuJoCo    bool
}

// LookupEnvInfo finds information about an environment.
func LookupEnvInfo(name string) (*EnvInfo, error) {
	spec := muniverse.SpecForName(name)
	if spec != nil {
		w, h := muniverseDownsampledSize(spec.Width, spec.Height)
		res := &EnvInfo{
			Name:          name,
			ActionSpace:   anyrl.Softmax{},
			ParamSize:     len(spec.KeyWhitelist) + 1,
			Width:         w,
			Height:        h,
			NumFeatures:   w * h,
			Uint8Features: true,
			Muniverse:     true,
		}
		if res.ParamSize == 1 {
			// Support tapping games with no keyboard.
			res.ActionSpace = &anyrl.Bernoulli{OneHot: true}
		}
		return res, nil
	}

	if info := cubeRLInfo(); name == info.Name {
		return cubeRLInfo(), nil
	}

	if numActions, ok := atariActionSizes[name]; ok {
		return &EnvInfo{
			Name:          name,
			ActionSpace:   anyrl.Softmax{},
			ParamSize:     numActions,
			Width:         atariWidth,
			Height:        atariHeight,
			NumFeatures:   atariObsSize(name),
			Uint8Features: true,
			Atari:         true,
		}, nil
	}

	if numActions, numObs, ok := mujocoEnvInfo(name); ok {
		return &EnvInfo{
			Name:        name,
			ActionSpace: anyrl.Gaussian{},
			ParamSize:   numActions * 2,
			NumFeatures: numObs,
			MuJoCo:      true,
		}, nil
	}

	return nil, errors.New("lookup game environment: \"" + name + "\" not found")
}

// EnvRoller creates a roller with the appropriate fields
// set for the environment.
func EnvRoller(c anyvec.Creator, e *EnvInfo, p *treeagent.Forest) *treeagent.Roller {
	roller := &treeagent.Roller{
		Policy:      p,
		ActionSpace: e.ActionSpace,
	}
	if e.Uint8Features {
		roller.MakeInputTape = func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		}
	}
	return roller
}

// EnvSamples optimizes the samples for the particular
// environment (e.g. with uint8 conversion).
func EnvSamples(e *EnvInfo, s <-chan treeagent.Sample) <-chan treeagent.Sample {
	if e.Uint8Features {
		return treeagent.Uint8Samples(s)
	}
	return s
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
	} else if info.CubeRL {
		return newCubeRLEnvs(c, e, n)
	} else if info.MuJoCo {
		return newMuJoCoEnvs(c, e, n)
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

	lastObs []float64
}

func (h *historyEnv) Reset() ([]float64, error) {
	obs, err := h.Env.Reset()
	h.lastObs = obs
	return h.nextObs(obs), err
}

func (h *historyEnv) Step(action []float64) ([]float64, float64, bool, error) {
	obs, rew, done, err := h.Env.Step(action)
	return h.nextObs(obs), rew, done, err
}

func (h *historyEnv) nextObs(obs []float64) []float64 {
	if obs == nil {
		return nil
	}
	res := append(append([]float64{}, obs...), h.lastObs...)
	h.lastObs = obs
	return res
}
