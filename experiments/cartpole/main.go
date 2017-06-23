// Trains a decision tree on CartPole-v0 in OpenAI Gym.

package main

import (
	"log"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	treeagent "github.com/unixpickle/treeagent"
	"github.com/unixpickle/weakai/idtrees"
)

const (
	Host             = "localhost:5001"
	RolloutsPerBatch = 50
	NumBatches       = 50
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

	// Create a forest-based policy which starts off
	// doing completely random actions.
	policy := &treeagent.Policy{
		Classifier: &treeagent.UniformClassifier{NumClasses: 2},
		NumActions: 2,
		Epsilon:    0.1,
	}

	// Setup a trainer.
	trainer := &treeagent.Trainer{
		NumTrees:    1,
		NumFeatures: 4,
		BuildTree: func(samples []idtrees.Sample, attrs []idtrees.Attr) *idtrees.Tree {
			return idtrees.LimitedID3(samples, attrs, 0, 4)
		},
	}

	// Setup a roller.
	roller := &treeagent.Roller{
		Policy:  policy,
		Creator: creator,
	}

	for batchIdx := 0; batchIdx < NumBatches; batchIdx++ {
		// Decrease exploration over time.
		policy.Epsilon *= 0.9

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
		policy.Classifier = trainer.Train(r)
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
