// Trains a decision tree on CartPole-v0 in OpenAI Gym.

package main

import (
	"log"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/treeagent"
)

const (
	Host             = "localhost:5001"
	RolloutsPerBatch = 100
	NumBatches       = 100

	StepSize  = 5
	StepDecay = 0.98
	Depth     = 6
)

func main() {
	// Connect to gym server.
	client, err := gym.Make(Host, "CartPole-v0")
	must(err)
	defer client.Close()

	// Create an anyrl.Env from our gym environment.
	env, err := anyrl.GymEnv(client, false)
	must(err)

	// Setup a roller with a zero-initialized policy.
	actionSpace := anyrl.Softmax{}
	roller := &treeagent.Roller{
		Policy:      treeagent.NewForest(2),
		ActionSpace: actionSpace,
	}

	// Setup a way to build trees.
	pg := &treeagent.PG{
		Builder: treeagent.Builder{
			MaxDepth:  Depth,
			Algorithm: treeagent.MSEAlgorithm,
		},
		ActionSpace: actionSpace,
	}

	var step float64 = StepSize
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
		judger := anypg.TotalJudger{Normalize: true}
		samples := treeagent.RolloutSamples(r, judger.JudgeActions(r))
		tree, _, _ := pg.Build(treeagent.AllSamples(samples))
		roller.Policy.Add(tree, step)
		step *= StepDecay
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
