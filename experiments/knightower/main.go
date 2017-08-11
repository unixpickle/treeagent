// Adapted from https://github.com/unixpickle/rl-agents/blob/7af9e208f3b5aa1d83abb266e7cd2ccf64b11ac2/knightower_tree/main.go

package main

import (
	"bytes"
	"compress/flate"
	"encoding/gob"
	"io/ioutil"
	"log"
	"math"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/treeagent"
)

const (
	ParallelEnvs = 4
	BatchSize    = 128
	LogInterval  = 16
	Depth        = 3
)

const (
	SaveFile = "trained_policy"
)

func main() {
	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Create a decision tree policy.
	policy := loadOrCreatePolicy()

	// Setup a Roller for producing rollouts.
	roller := &treeagent.Roller{
		Policy:     policy,
		Creator:    creator,
		NumActions: 2,

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	// Setup a trainer for producing new policies.
	trainer := &treeagent.Trainer{
		StepSize:     0.8,
		TrainingMode: treeagent.LinearUpdate,
	}

	// Train on a background goroutine so that we can
	// listen for Ctrl+C on the main goroutine.
	var trainLock sync.Mutex
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			// Join the rollouts into one set.
			rollouts := gatherRollouts(roller)
			r := anyrl.PackRolloutSets(rollouts)

			// Print the stats for the batch.
			log.Printf("batch %d: mean=%f stddev=%f", batchIdx,
				r.Rewards.Mean(), math.Sqrt(r.Rewards.Variance()))

			// Train on the rollouts.
			log.Println("Training on batch...")
			samples := treeagent.Uint8Samples(NumFeatures, treeagent.RolloutSamples(r))
			targets := trainer.Targets(r, samples)
			policy = treeagent.BuildTree(treeagent.AllSamples(targets), NumFeatures, Depth)
			roller.Policy = policy

			// Save the new policy.
			trainLock.Lock()
			var data bytes.Buffer
			enc := gob.NewEncoder(&data)
			must(enc.Encode(policy))
			must(ioutil.WriteFile(SaveFile, data.Bytes(), 0755))
			trainLock.Unlock()
		}
	}()

	log.Println("Running. Press Ctrl+C to stop.")
	<-rip.NewRIP().Chan()

	// Avoid the race condition where we save during
	// exit.
	trainLock.Lock()
}

func gatherRollouts(roller *treeagent.Roller) []*anyrl.RolloutSet {
	resChan := make(chan *anyrl.RolloutSet, BatchSize)

	requests := make(chan struct{}, BatchSize)
	for i := 0; i < BatchSize; i++ {
		requests <- struct{}{}
	}
	close(requests)

	var wg sync.WaitGroup
	for i := 0; i < ParallelEnvs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			spec := muniverse.SpecForName("Knightower-v0")
			if spec == nil {
				panic("environment not found")
			}
			env, err := muniverse.NewEnv(spec)

			// Used to debug on my end.
			//env, err := muniverse.NewEnvChrome("localhost:9222", "localhost:8080", spec)

			must(err)
			defer env.Close()

			preproc := &Env{
				Env:     env,
				Creator: roller.Creator,
			}
			for _ = range requests {
				rollout, err := roller.Rollout(preproc)
				must(err)
				resChan <- rollout
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resChan)
	}()

	var res []*anyrl.RolloutSet
	var batchRewardSum float64
	var numBatchReward int
	for item := range resChan {
		res = append(res, item)
		numBatchReward++
		batchRewardSum += item.Rewards.Mean()
		if numBatchReward == LogInterval || len(res) == BatchSize {
			log.Printf("sub_mean=%f", batchRewardSum/float64(numBatchReward))
			numBatchReward = 0
			batchRewardSum = 0
		}
	}
	return res
}

func loadOrCreatePolicy() *treeagent.Tree {
	data, err := ioutil.ReadFile(SaveFile)
	if err != nil {
		log.Println("Created new policy.")
		return &treeagent.Tree{Distribution: treeagent.NewActionDist(2)}
	}
	var res *treeagent.Tree
	dec := gob.NewDecoder(bytes.NewReader(data))
	must(dec.Decode(&res))
	log.Println("Loaded policy from file.")
	return res
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
