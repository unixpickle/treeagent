package experiments

import (
	"errors"
	"time"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
)

const muniverseDownsample = 4

// muniverseEnv is an anyrl.Env wrapper around a
// muniverse.Env.
// It handles action conversions and downsampling.
type muniverseEnv struct {
	Env         muniverse.Env
	Creator     anyvec.Creator
	TimePerStep time.Duration

	timestep int
}

// newMuniverseEnvs creates n environment instances.
func newMuniverseEnvs(c anyvec.Creator, g *GameFlags, n int) ([]Env, error) {
	spec := muniverse.SpecForName(g.Name)
	if spec == nil {
		return nil, errors.New(`"` + g.Name + `" not found`)
	}

	var res []Env
	for i := 0; i < n; i++ {
		env, err := muniverse.NewEnv(spec)
		if err != nil {
			CloseEnvs(res)
			return nil, err
		}

		if g.RecordDir != "" {
			env = muniverse.RecordEnv(env, g.RecordDir)
		}

		res = append(res, &muniverseEnv{
			Env:         env,
			Creator:     c,
			TimePerStep: g.FrameTime,
		})
	}

	return res, nil
}

// Reset sets up a fresh instance of the environment.
func (m *muniverseEnv) Reset() (observation anyvec.Vector, err error) {
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
func (m *muniverseEnv) Step(action anyvec.Vector) (observation anyvec.Vector,
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

// Close shuts down the environment.
func (m *muniverseEnv) Close() error {
	return m.Env.Close()
}

func (m *muniverseEnv) simplifyImage(in []uint8) anyvec.Vector {
	spec := m.Env.Spec()
	w, h := muniverseDownsampledSize(spec.Width, spec.Height)
	data := make([]float64, 0, w*h)
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

func muniverseDownsampledSize(width, height int) (int, int) {
	subWidth := width / 4
	subHeight := height / 4
	if width%4 != 0 {
		subWidth++
	}
	if height%4 != 0 {
		subHeight++
	}
	return subWidth, subHeight
}
