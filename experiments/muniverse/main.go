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
	"github.com/unixpickle/anyrl/anypg"
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
	Discount     float64
	SaveFile     string
	Env          string
	RecordDir    string
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
	flag.Float64Var(&flags.Discount, "discount", 0, "discount factor (0 is no discount)")
	flag.StringVar(&flags.SaveFile, "out", "policy.json", "file for saved policy")
	flag.StringVar(&flags.Env, "env", "", "environment (e.g. Knightower-v0)")
	flag.StringVar(&flags.RecordDir, "record", "", "directory to save recordings")
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

	var judger anypg.ActionJudger
	if flags.Discount != 0 {
		judger = &anypg.QJudger{Discount: flags.Discount, Normalize: true}
	} else {
		judger = &anypg.TotalJudger{Normalize: true}
	}

	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Setup a Roller for producing rollouts.
	roller := &treeagent.Roller{
		Policy:      loadOrCreatePolicy(flags),
		Creator:     creator,
		ActionSpace: anyrl.Softmax{},

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
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
			advantages := judger.JudgeActions(r)
			rawSamples := treeagent.RolloutSamples(r, advantages)
			samples := treeagent.Uint8Samples(numFeatures, rawSamples)
			tree := treeagent.BuildTree(treeagent.AllSamples(samples),
				anyrl.Softmax{}, numFeatures, flags.Depth)
			roller.Policy.Add(tree, flags.StepSize)

			// Save the new policy.
			trainLock.Lock()
			data, err := json.Marshal(roller.Policy)
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
			spec := muniverse.SpecForName(flags.Env)
			if spec == nil {
				panic("environment not found")
			}
			env, err := muniverse.NewEnv(spec)
			must(err)

			if flags.RecordDir != "" {
				env = muniverse.RecordEnv(env, flags.RecordDir)
			}

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

func loadOrCreatePolicy(flags *Flags) *treeagent.Forest {
	data, err := ioutil.ReadFile(flags.SaveFile)
	if err != nil {
		log.Println("Created new policy.")
		n := 1 + len(muniverse.SpecForName(flags.Env).KeyWhitelist)
		return treeagent.NewForest(n)
	}
	var res *treeagent.Forest
	must(json.Unmarshal(data, &res))
	log.Println("Loaded policy from file.")
	return res
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
