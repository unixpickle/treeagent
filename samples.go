package treeagent

import (
	"math"

	"github.com/unixpickle/anyrl"
)

// Sample is a training sample for building a tree.
//
// Each Sample associates an observation with a target
// action distribution.
// During training, Samples can be constructed with a
// similar action distribution to the original, but with a
// slight modification in the direction of improvement.
type Sample interface {
	Feature(idx int) float64
	ActionDist() ActionDist
}

// RolloutSamples uses the input and output tapes in a set
// of rollouts to produce a stream of Samples.
// Each Sample represents a single timestep and the action
// distribution matches the policy's output.
//
// The resulting channel is sorted first by timestep and
// then by index in the batch.
// Thus, Samples from time t are always before Samples
// from time t+1.
//
// The caller must read the entire channel to prevent a
// resource leak.
func RolloutSamples(r *anyrl.RolloutSet) <-chan Sample {
	res := make(chan Sample, 1)
	go func() {
		defer close(res)
		inChan := r.Inputs.ReadTape(0, -1)
		outChan := r.AgentOuts.ReadTape(0, -1)
		for {
			input := <-inChan
			output := <-outChan
			inValues := vecToFloats(input.Packed)
			outValues := vecToFloats(output.Packed)

			batch := input.NumPresent()
			numFeatures := len(inValues) / batch
			numActions := len(outValues) / batch
			for i := 0; i < batch; i++ {
				subIns := inValues[i*numFeatures : (i+1)*numFeatures]
				subOuts := outValues[i*numActions : (i+1)*numActions]
				actionDist := ActionDist{}
				for action, logProb := range subOuts {
					if !math.IsInf(logProb, -1) {
						actionDist[Action(action)] = math.Exp(logProb)
					}
				}
				res <- &memorySample{
					Features: subIns,
					Dist:     actionDist,
				}
			}
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
func Uint8Samples(numFeatures int, incoming <-chan Sample) <-chan Sample {
	res := make(chan Sample, 1)
	go func() {
		for in := range incoming {
			sample := &uint8Sample{
				Features: make([]uint8, numFeatures),
				Dist:     in.ActionDist(),
			}
			for i := 0; i < numFeatures; i++ {
				sample.Features[i] = uint8(in.Feature(i))
			}
			res <- sample
		}
	}()
	return res
}

type memorySample struct {
	Features []float64
	Dist     ActionDist
}

func (m *memorySample) Feature(idx int) float64 {
	return m.Features[idx]
}

func (m *memorySample) ActionDist() ActionDist {
	return m.Dist
}

type uint8Sample struct {
	Features []uint8
	Dist     ActionDist
}

func (u *uint8Sample) Feature(idx int) float64 {
	return float64(u.Features[idx])
}

func (u *uint8Sample) ActionDist() ActionDist {
	return u.Dist
}
