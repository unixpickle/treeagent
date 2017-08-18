package main

import (
	"time"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
)

// NumFeatures returns the number of observation features
// for the environment (after downsampling).
func NumFeatures(e *muniverse.EnvSpec) int {
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

// Env is an anyrl.Env wrapper around a muniverse.Env.
//
// Action vectors are one-hot vectors indicating which key
// to press at each timestep.
// No key holding is performed.
type Env struct {
	Env         muniverse.Env
	Creator     anyvec.Creator
	TimePerStep time.Duration

	timestep int
}

func (e *Env) Reset() (observation anyvec.Vector, err error) {
	err = e.Env.Reset()
	if err != nil {
		return
	}
	rawObs, err := e.Env.Observe()
	if err != nil {
		return
	}
	buffer, _, _, err := muniverse.RGB(rawObs)
	if err != nil {
		return
	}
	observation = e.simplifyImage(buffer)
	e.timestep = 0
	return
}

func (e *Env) Step(action anyvec.Vector) (observation anyvec.Vector,
	reward float64, done bool, err error) {
	var events []interface{}
	actionIdx := anyvec.MaxIndex(action)
	actions := append([]string{""}, e.Env.Spec().KeyWhitelist...)
	actionKey := actions[actionIdx]
	if actionKey != "" {
		evt := chrome.KeyEvents[actionKey]
		evt1 := evt
		evt.Type = chrome.KeyDown
		evt1.Type = chrome.KeyUp
		events = append(events, &evt, &evt1)
	}

	reward, done, err = e.Env.Step(e.TimePerStep, events...)
	if err != nil {
		return
	}
	rawObs, err := e.Env.Observe()
	if err != nil {
		return
	}
	buffer, _, _, err := muniverse.RGB(rawObs)
	if err != nil {
		return
	}
	observation = e.simplifyImage(buffer)

	if time.Duration(e.timestep)*e.TimePerStep >= time.Minute {
		done = true
	}
	return
}

func (e *Env) simplifyImage(in []uint8) anyvec.Vector {
	spec := e.Env.Spec()
	data := make([]float64, 0, NumFeatures(spec))
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
	return e.Creator.MakeVectorData(e.Creator.MakeNumericList(data))
}
