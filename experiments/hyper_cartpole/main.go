package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
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
	fmt.Println("depth,step_size,step_decay,reward")

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
	depth := rand.Intn(8)
	stepSize := math.Exp(rand.Float64()*8 - 4)
	stepDecay := rand.Float64()

	// Setup a roller with a uniformly random policy.
	roller := &treeagent.Roller{
		Policy:      treeagent.NewForest(2),
		Creator:     creator,
		ActionSpace: anyrl.Softmax{},
	}

	var lastMean float64
	currentStep := stepSize
	for batchIdx := 0; batchIdx <= NumBatches; batchIdx++ {
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

		if batchIdx == NumBatches {
			break
		}

		// Train on the rollouts.
		judger := anypg.TotalJudger{Normalize: true}
		samples := treeagent.RolloutSamples(r, judger.JudgeActions(r))
		tree := treeagent.BuildTree(treeagent.AllSamples(samples), anyrl.Softmax{},
			4, depth)
		roller.Policy.Add(tree, currentStep)
		currentStep *= stepDecay
	}

	fmt.Printf("%d,%f,%f,%f\n", depth, stepSize, stepDecay, lastMean)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
