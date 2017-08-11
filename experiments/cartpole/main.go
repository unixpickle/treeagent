// Trains a decision tree on CartPole-v0 in OpenAI Gym.

package main

import (
	"log"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/treeagent"
)

const (
	Host             = "localhost:5001"
	RolloutsPerBatch = 100
	NumBatches       = 100

	Depth = 3
)

func main() {
	// Used to create vectors.
	creator := anyvec32.CurrentCreator()

	// Connect to gym server.
	client, err := gym.Make(Host, "CartPole-v0")
	must(err)
	defer client.Close()

	// Create an anyrl.Env from our gym environment.
	env, err := anyrl.GymEnv(creator, client, false)
	must(err)

	// Setup a trainer.
	trainer := &treeagent.Trainer{
		StepSize:     0.8,
		TrainingMode: treeagent.LinearUpdate,
	}

	// Setup a roller with a uniformly random policy.
	roller := &treeagent.Roller{
		Policy:     &treeagent.Tree{Distribution: treeagent.NewActionDist(2)},
		Creator:    creator,
		NumActions: 2,
	}

	for batchIdx := 0; batchIdx < NumBatches; batchIdx++ {
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
		log.Printf("batch %d: mean_reward=%f", batchIdx, r.Rewards.Mean())

		// Train on the rollouts.
		samples := treeagent.RolloutSamples(r)
		targets := trainer.Targets(r, samples)
		roller.Policy = treeagent.BuildTree(treeagent.AllSamples(targets), 4, Depth)
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
