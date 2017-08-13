package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/treeagent"
)

const (
	Host             = "localhost:5001"
	RolloutsPerBatch = 100
	NumBatches       = 30
)

func main() {
	fmt.Println("step_size,entropy_reg,depth,step_decay,reward")

	creator := anyvec32.CurrentCreator()

	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		client, err := gym.Make(Host, "CartPole-v0")
		must(err)
		defer client.Close()

		env, err := anyrl.GymEnv(creator, client, false)
		must(err)
		go func() {
			for {
				randomTrainingRound(creator, env)
			}
		}()
	}
	select {}
}

func randomTrainingRound(creator anyvec.Creator, env anyrl.Env) {
	stepSize := math.Exp(rand.Float64()*5 - 4)
	entropyReg := math.Exp(rand.Float64()*5 - 4)
	stepDecay := 1 - rand.Float64()*0.1
	depth := rand.Intn(8)

	trainer := &treeagent.Trainer{
		StepSize:     stepSize,
		EntropyReg:   entropyReg,
		TrainingMode: treeagent.LinearUpdate,
	}

	// Setup a roller with a uniformly random policy.
	roller := &treeagent.Roller{
		Policy:  &treeagent.Tree{Distribution: treeagent.NewActionDist(2)},
		Creator: creator,
	}

	var lastMean float64
	for batchIdx := 0; batchIdx < NumBatches; batchIdx++ {
		trainer.StepSize *= stepDecay

		// Gather episode rollouts.
		var rollouts []*anyrl.RolloutSet
		for i := 0; i < RolloutsPerBatch; i++ {
			rollout, err := roller.Rollout(env)
			must(err)
			rollouts = append(rollouts, rollout)
		}

		// Join the rollouts into one set.
		r := anyrl.PackRolloutSets(rollouts)

		// Print the rewards.
		lastMean = r.Rewards.Mean()

		// Train on the rollouts.
		samples := treeagent.RolloutSamples(r)
		targets := trainer.Targets(r, samples)
		roller.Policy = treeagent.BuildTree(treeagent.AllSamples(targets), 4, depth)
	}

	fmt.Printf("%f,%f,%d,%f,%f\n", stepSize, entropyReg, depth, stepDecay, lastMean)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
