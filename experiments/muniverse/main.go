package main

import (
	"compress/flate"
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"math"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/treeagent"
)

type Flags struct {
	BatchSize    int
	ParallelEnvs int
	LogInterval  int
	Depth        int
	StepSize     float64
	SaveFile     string
	Env          string
	FrameTime    time.Duration
}

func main() {
	flags := &Flags{}
	flag.IntVar(&flags.BatchSize, "batch", 128, "rollout batch size")
	flag.IntVar(&flags.ParallelEnvs, "numparallel", runtime.GOMAXPROCS(0),
		"parallel environments")
	flag.IntVar(&flags.LogInterval, "logint", 16, "episodes per log")
	flag.IntVar(&flags.Depth, "depth", 3, "tree depth")
	flag.Float64Var(&flags.StepSize, "step", 0.8, "step size")
	flag.StringVar(&flags.SaveFile, "out", "policy.json", "file for saved policy")
	flag.StringVar(&flags.Env, "env", "", "environment (e.g. Knightower-v0)")
	flag.DurationVar(&flags.FrameTime, "frametime", time.Second/8, "time per frame")
	flag.Parse()

	if flags.Env == "" {
		essentials.Die("Missing -env flag. See -help.")
	}
	spec := muniverse.SpecForName(flags.Env)
	if spec == nil {
		essentials.Die("unknown environment:", flags.Env)
	}

	log.Println("Run with arguments:", os.Args[1:])

	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Create a decision tree policy.
	policy := loadOrCreatePolicy(flags)

	// Setup a Roller for producing rollouts.
	roller := &treeagent.Roller{
		Policy:  policy,
		Creator: creator,

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	// Setup a trainer for producing new policies.
	trainer := &treeagent.Trainer{
		StepSize:     flags.StepSize,
		TrainingMode: treeagent.LinearUpdate,
	}

	// Train on a background goroutine so that we can
	// listen for Ctrl+C on the main goroutine.
	var trainLock sync.Mutex
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			// Join the rollouts into one set.
			rollouts := gatherRollouts(flags, roller)
			r := anyrl.PackRolloutSets(rollouts)

			// Print the stats for the batch.
			log.Printf("batch %d: mean=%f stddev=%f", batchIdx,
				r.Rewards.Mean(), math.Sqrt(r.Rewards.Variance()))

			// Train on the rollouts.
			log.Println("Training on batch...")
			numFeatures := NumFeatures(spec)
			samples := treeagent.Uint8Samples(numFeatures, treeagent.RolloutSamples(r))
			targets := trainer.Targets(r, samples)
			policy = treeagent.BuildTree(treeagent.AllSamples(targets), numFeatures,
				flags.Depth)
			roller.Policy = policy

			// Save the new policy.
			trainLock.Lock()
			data, err := json.Marshal(policy)
			must(err)
			must(ioutil.WriteFile(flags.SaveFile, data, 0755))
			trainLock.Unlock()
		}
	}()

	log.Println("Running. Press Ctrl+C to stop.")
	<-rip.NewRIP().Chan()

	// Avoid the race condition where we save during
	// exit.
	trainLock.Lock()
}

func gatherRollouts(flags *Flags, roller *treeagent.Roller) []*anyrl.RolloutSet {
	resChan := make(chan *anyrl.RolloutSet, flags.BatchSize)

	requests := make(chan struct{}, flags.BatchSize)
	for i := 0; i < flags.BatchSize; i++ {
		requests <- struct{}{}
	}
	close(requests)

	var wg sync.WaitGroup
	for i := 0; i < flags.ParallelEnvs; i++ {
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
				Env:         env,
				Creator:     roller.Creator,
				TimePerStep: flags.FrameTime,
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
		if numBatchReward == flags.LogInterval || len(res) == flags.BatchSize {
			log.Printf("sub_mean=%f", batchRewardSum/float64(numBatchReward))
			numBatchReward = 0
			batchRewardSum = 0
		}
	}
	return res
}

func loadOrCreatePolicy(path string) *treeagent.Tree {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		log.Println("Created new policy.")
		return &treeagent.Tree{Distribution: treeagent.NewActionDist(2)}
	}
	var res *treeagent.Tree
	must(json.Unmarshal(data, &res))
	log.Println("Loaded policy from file.")
	return res
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
