// Taken from https://github.com/unixpickle/rl-agents/blob/7af9e208f3b5aa1d83abb266e7cd2ccf64b11ac2/knightower_tree/preprocess.go.

package main

import (
	"time"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
)

const (
	FrameWidth  = 320
	FrameHeight = 480

	MaxTimestep = 480
	TimePerStep = time.Second / 8

	NumFeatures = (FrameWidth / 4) * (FrameHeight / 4)
)

type Env struct {
	Env     muniverse.Env
	Creator anyvec.Creator

	Timestep int
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
	e.Timestep = 0
	return
}

func (e *Env) Step(action anyvec.Vector) (observation anyvec.Vector,
	reward float64, done bool, err error) {
	var events []interface{}
	actionIdx := anyvec.MaxIndex(action)
	key := []string{"ArrowLeft", "ArrowRight", ""}[actionIdx]
	if key != "" {
		evt := chrome.KeyEvents[key]
		evt1 := evt
		evt.Type = chrome.KeyDown
		evt1.Type = chrome.KeyUp
		events = append(events, &evt, &evt1)
	}

	reward, done, err = e.Env.Step(TimePerStep, events...)
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

	e.Timestep++
	if e.Timestep > MaxTimestep {
		done = true
	}
	return
}

func (e *Env) simplifyImage(in []uint8) anyvec.Vector {
	data := make([]float64, 0, NumFeatures)
	for y := 0; y < FrameHeight; y += 4 {
		for x := 0; x < FrameWidth; x += 4 {
			sourceIdx := (y*FrameWidth + x) * 3
			var value float64
			for d := 0; d < 3; d++ {
				value += float64(in[sourceIdx+d])
			}
			data = append(data, essentials.Round(value/3))
		}
	}
	return e.Creator.MakeVectorData(e.Creator.MakeNumericList(data))
}
