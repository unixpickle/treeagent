package treeagent

import (
	"math"
	"math/rand"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
)

// A FeatureSource is a list of numerical features.
type FeatureSource interface {
	Feature(idx int) float64
}

// Sample is a training sample for building a tree.
//
// Each Sample provides information about a single
// timestep in an episode.
type Sample interface {
	FeatureSource
	NumFeatures() int
	Action() anyvec.Vector
	ActionParams() anyvec.Vector
	Advantage() float64
}

// RolloutSamples produces a stream of Samples based on
// the batch of rollouts.
// The advantages can come from an anypg.ActionJudger.
//
// The resulting channel is sorted first by timestep and
// then by index in the batch.
// Thus, Samples from time t are always before Samples
// from time t+1.
//
// The caller must read the entire channel to prevent a
// resource leak.
func RolloutSamples(r *anyrl.RolloutSet, advantages anyrl.Rewards) <-chan Sample {
	res := make(chan Sample, 1)
	go func() {
		defer close(res)
		inChan := r.Inputs.ReadTape(0, -1)
		actChan := r.Actions.ReadTape(0, -1)
		outChan := r.AgentOuts.ReadTape(0, -1)
		timestep := 0
		for inputs := range inChan {
			outputs := <-outChan
			actions := <-actChan
			inValues := vecToFloats(inputs.Packed)

			batch := inputs.NumPresent()
			numFeatures := len(inValues) / batch
			actSize := actions.Packed.Len() / batch
			outSize := outputs.Packed.Len() / batch
			i := 0
			for lane, pres := range inputs.Present {
				if !pres {
					continue
				}
				subIns := inValues[i*numFeatures : (i+1)*numFeatures]
				subActs := actions.Packed.Slice(i*actSize, (i+1)*actSize)
				subOuts := outputs.Packed.Slice(i*outSize, (i+1)*outSize)
				res <- &memorySample{
					features:     subIns,
					action:       subActs,
					actionParams: subOuts,
					advantage:    advantages[lane][timestep],
				}
				i++
			}
			timestep++
		}
	}()
	return res
}

// Uint8Samples shrinks the memory footprint of a Sample
// stream by storing the features as uint8 values.
//
// The order of the input channel is preserved in the
// output channel.
//
// You should only use this if you know that the features
// are 8-bit integers.
//
// The caller must read the entire channel to prevent a
// resource leak.
// Doing so will automatically read the incoming channel
// in its entirety.
func Uint8Samples(incoming <-chan Sample) <-chan Sample {
	res := make(chan Sample, 1)
	go func() {
		defer close(res)
		for in := range incoming {
			dim := in.NumFeatures()
			sample := &uint8Sample{
				features:     make([]uint8, dim),
				action:       in.Action(),
				actionParams: in.ActionParams(),
				advantage:    in.Advantage(),
			}
			for i := 0; i < dim; i++ {
				sample.features[i] = uint8(in.Feature(i))
			}
			res <- sample
		}
	}()
	return res
}

// AllSamples reads the samples from the channel and
// stores them in a slice.
func AllSamples(ch <-chan Sample) []Sample {
	var res []Sample
	for s := range ch {
		res = append(res, s)
	}
	return res
}

// Minibatch selects a random fraction of the samples.
func Minibatch(samples []Sample, frac float64) []Sample {
	count := int(math.Ceil(float64(len(samples)) * frac))
	if count == 0 {
		count = len(samples)
	}
	res := make([]Sample, count)
	for i, j := range rand.Perm(len(samples))[:count] {
		res[i] = samples[j]
	}
	return res
}

type memorySample struct {
	features     []float64
	action       anyvec.Vector
	actionParams anyvec.Vector
	advantage    float64
}

func (m *memorySample) Feature(idx int) float64 {
	return m.features[idx]
}

func (m *memorySample) NumFeatures() int {
	return len(m.features)
}

func (m *memorySample) Action() anyvec.Vector {
	return m.action
}

func (m *memorySample) ActionParams() anyvec.Vector {
	return m.actionParams
}

func (m *memorySample) Advantage() float64 {
	return m.advantage
}

type uint8Sample struct {
	features     []uint8
	action       anyvec.Vector
	actionParams anyvec.Vector
	advantage    float64
}

func (u *uint8Sample) Feature(idx int) float64 {
	return float64(u.features[idx])
}

func (u *uint8Sample) NumFeatures() int {
	return len(u.features)
}

func (u *uint8Sample) Action() anyvec.Vector {
	return u.action
}

func (u *uint8Sample) ActionParams() anyvec.Vector {
	return u.actionParams
}

func (u *uint8Sample) Advantage() float64 {
	return u.advantage
}
