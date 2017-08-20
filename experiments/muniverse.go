package experiments

import (
	"errors"
	"sync"
	"time"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
	"github.com/unixpickle/treeagent"
)

type ActionSpace interface {
	anyrl.LogProber
	anyrl.Sampler
	anyrl.Entropyer
}

// NumFeaturesMuniverse returns the number of observation
// features for the environment (after downsampling).
func NumFeaturesMuniverse(e *muniverse.EnvSpec) int {
	width := e.Width / 4
	height := e.Height / 4
	if e.Width%4 != 0 {
		width++
	}
	if e.Height%4 != 0 {
		height++
	}
	return width * height
}

// ActionSpaceMuniverse returns the action space and the
// number of actions for a muniverse environment.
func ActionSpaceMuniverse(e *muniverse.EnvSpec) (ActionSpace, int) {
	return anyrl.Softmax{}, len(e.KeyWhitelist) + 1
}

// GatherRolloutsMuniverse produces rollouts using the
// provided environments.
//
// The steps argument specifies the minimum number of
// timesteps in the resulting rollouts.
//
// Along with the rollouts, it produces an entropy measure
// to indicate how much exploration took place.
func GatherRolloutsMuniverse(roller *treeagent.Roller, envs []*MuniverseEnv,
	steps int) (*anyrl.RolloutSet, anyvec.Numeric, error) {
	resChan := make(chan *anyrl.RolloutSet, 1)
	errChan := make(chan error, 1)
	requests := make(chan struct{}, len(envs))
	for i := 0; i < len(envs); i++ {
		requests <- struct{}{}
	}

	var wg sync.WaitGroup
	for _, env := range envs {
		wg.Add(1)
		go func(env anyrl.Env) {
			defer wg.Done()
			for _ = range requests {
				rollout, err := roller.Rollout(env)
				if err != nil {
					select {
					case errChan <- err:
					default:
					}
					return
				}
				resChan <- rollout
			}
		}(env)
	}

	go func() {
		wg.Wait()
		close(resChan)
		close(errChan)
	}()

	var res []*anyrl.RolloutSet
	var totalSteps int
	for item := range resChan {
		res = append(res, item)
		if totalSteps < steps {
			totalSteps += item.NumSteps()
			if totalSteps < steps {
				requests <- struct{}{}
			} else {
				close(requests)
			}
		}
	}
	packed := anyrl.PackRolloutSets(res)

	reg := &anypg.EntropyReg{
		Entropyer: roller.ActionSpace.(anyrl.Entropyer),
		Coeff:     1,
	}
	entropy := anypg.AverageReg(roller.Creator, packed.AgentOuts, reg)

	return packed, entropy, <-errChan
}

// MuniverseEnv is an anyrl.Env wrapper around a
// muniverse.Env.
// It handles action conversions and downsampling.
//
// Action vectors are one-hot vectors indicating which key
// to press at each timestep.
// No key holding is performed.
type MuniverseEnv struct {
	Env         muniverse.Env
	Creator     anyvec.Creator
	TimePerStep time.Duration

	timestep int
}

// NewMuniverseEnvs creates n environment instances.
func NewMuniverseEnvs(c anyvec.Creator, f *MuniverseEnvFlags,
	n int) ([]*MuniverseEnv, error) {
	spec := muniverse.SpecForName(f.Name)
	if spec == nil {
		return nil, errors.New("create environments: no environment named " + f.Name)
	}

	var res []*MuniverseEnv
	for i := 0; i < n; i++ {
		env, err := muniverse.NewEnv(spec)
		if err != nil {
			for _, e := range res {
				e.Env.Close()
			}
			return nil, err
		}

		if f.RecordDir != "" {
			env = muniverse.RecordEnv(env, f.RecordDir)
		}

		res = append(res, &MuniverseEnv{
			Env:         env,
			Creator:     c,
			TimePerStep: f.FrameTime,
		})
	}

	return res, nil
}

// Reset sets up a fresh instance of the environment.
func (m *MuniverseEnv) Reset() (observation anyvec.Vector, err error) {
	err = m.Env.Reset()
	if err != nil {
		return
	}
	rawObs, err := m.Env.Observe()
	if err != nil {
		return
	}
	buffer, _, _, err := muniverse.RGB(rawObs)
	if err != nil {
		return
	}
	observation = m.simplifyImage(buffer)
	m.timestep = 0
	return
}

// Step takes an action, advances time, and captures a
// screenshot of the environment.
func (m *MuniverseEnv) Step(action anyvec.Vector) (observation anyvec.Vector,
	reward float64, done bool, err error) {
	var events []interface{}
	actionIdx := anyvec.MaxIndex(action)
	actions := append([]string{""}, m.Env.Spec().KeyWhitelist...)
	actionKey := actions[actionIdx]
	if actionKey != "" {
		evt := chrome.KeyEvents[actionKey]
		evt1 := evt
		evt.Type = chrome.KeyDown
		evt1.Type = chrome.KeyUp
		events = append(events, &evt, &evt1)
	}

	reward, done, err = m.Env.Step(m.TimePerStep, events...)
	if err != nil {
		return
	}
	rawObs, err := m.Env.Observe()
	if err != nil {
		return
	}
	buffer, _, _, err := muniverse.RGB(rawObs)
	if err != nil {
		return
	}
	observation = m.simplifyImage(buffer)

	if time.Duration(m.timestep)*m.TimePerStep >= time.Minute {
		done = true
	}
	return
}

func (m *MuniverseEnv) simplifyImage(in []uint8) anyvec.Vector {
	spec := m.Env.Spec()
	data := make([]float64, 0, NumFeaturesMuniverse(spec))
	for y := 0; y < spec.Height; y += 4 {
		for x := 0; x < spec.Width; x += 4 {
			sourceIdx := (y*spec.Width + x) * 3
			var value float64
			for d := 0; d < 3; d++ {
				value += float64(in[sourceIdx+d])
			}
			data = append(data, essentials.Round(value/3))
		}
	}
	return m.Creator.MakeVectorData(m.Creator.MakeNumericList(data))
}
