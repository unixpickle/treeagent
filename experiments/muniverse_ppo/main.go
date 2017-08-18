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
	"github.com/unixpickle/anyvec"
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

	Depth      int
	StepSize   float64
	Discount   float64
	Lambda     float64
	EntropyReg float64
	Epsilon    float64
	Iters      int

	Algorithm string

	ActorFile  string
	CriticFile string

	Env       string
	RecordDir string
	FrameTime time.Duration
}

func main() {
	flags := &Flags{}
	flag.IntVar(&flags.BatchSize, "batch", 128, "rollout batch size")
	flag.IntVar(&flags.ParallelEnvs, "numparallel", runtime.GOMAXPROCS(0),
		"parallel environments")
	flag.IntVar(&flags.LogInterval, "logint", 16, "episodes per log")
	flag.IntVar(&flags.Depth, "depth", 8, "tree depth")
	flag.Float64Var(&flags.StepSize, "step", 0.8, "step size")
	flag.Float64Var(&flags.Discount, "discount", 0.8, "discount factor")
	flag.Float64Var(&flags.Lambda, "lambda", 0.95, "GAE coefficient")
	flag.Float64Var(&flags.EntropyReg, "reg", 0.01, "entropy regularization coefficient")
	flag.Float64Var(&flags.Epsilon, "epsilon", 0.1, "PPO epsilon")
	flag.IntVar(&flags.Iters, "iters", 4, "training iterations per batch")
	flag.StringVar(&flags.Algorithm, "algo", "mse", "tree algorithm ('mse' or 'sum')")
	flag.StringVar(&flags.ActorFile, "actor", "actor.json", "file for saved policy")
	flag.StringVar(&flags.CriticFile, "critic", "critic.json", "file for saved value function")
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

	creator := anyvec32.CurrentCreator()
	actionSpace := anyrl.Softmax{}

	policy, valueFunc := loadOrCreateForests(flags)

	roller := &treeagent.Roller{
		Policy:      policy,
		Creator:     creator,
		ActionSpace: actionSpace,

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	judger := &treeagent.Judger{
		ValueFunc: valueFunc,
		Discount:  flags.Discount,
		Lambda:    flags.Lambda,
	}

	ppo := &treeagent.PPO{
		Builder: &treeagent.Builder{
			NumFeatures: NumFeatures(spec),
			MaxDepth:    flags.Depth,
			ActionSpace: actionSpace,
			Regularizer: &anypg.EntropyReg{
				Entropyer: actionSpace,
				Coeff:     flags.EntropyReg,
			},
		},
	}

	switch flags.Algorithm {
	case "mse":
		ppo.Builder.Algorithm = treeagent.MSEAlgorithm
	case "sum":
		ppo.Builder.Algorithm = treeagent.SumAlgorithm
	default:
		essentials.Die("unknown algorithm:", flags.Algorithm)
	}

	var trainLock sync.Mutex
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			rollouts := gatherRollouts(flags, roller)
			r := anyrl.PackRolloutSets(rollouts)

			log.Printf("batch %d: mean=%f stddev=%f entropy=%f", batchIdx,
				r.Rewards.Mean(), math.Sqrt(r.Rewards.Variance()),
				actionEntropy(creator, r))

			numFeatures := NumFeatures(spec)

			log.Println("Training policy...")
			advantages := judger.JudgeActions(r)
			rawSamples := treeagent.RolloutSamples(r, advantages)
			sampleChan := treeagent.Uint8Samples(numFeatures, rawSamples)
			samples := treeagent.AllSamples(sampleChan)
			for i := 0; i < flags.Iters; i++ {
				tree, obj := ppo.Step(samples, policy)
				log.Printf("step %d: objective=%v", i, obj)
				policy.Add(tree, flags.StepSize)
			}

			log.Println("Training value function...")
			for i := 0; i < flags.Iters; i++ {
				advSamples := judger.TrainingSamples(r)
				sampleChan := treeagent.Uint8Samples(numFeatures, advSamples)
				samples := treeagent.AllSamples(sampleChan)

				var totalError float64
				for _, sample := range samples {
					totalError += math.Pow(sample.Advantage(), 2)
				}
				log.Printf("step %d: mse=%f", i, totalError/float64(len(samples)))

				judger.Train(samples, numFeatures, flags.Depth, flags.StepSize)
			}

			log.Println("Saving...")
			trainLock.Lock()

			data, err := json.Marshal(policy)
			must(err)
			must(ioutil.WriteFile(flags.ActorFile, data, 0755))

			data, err = json.Marshal(valueFunc)
			must(err)
			must(ioutil.WriteFile(flags.CriticFile, data, 0755))

			trainLock.Unlock()
		}
	}()

	log.Println("Running. Press Ctrl+C to stop.")
	<-rip.NewRIP().Chan()

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

func loadOrCreateForests(flags *Flags) (actor, critic *treeagent.Forest) {
	numActions := 1 + len(muniverse.SpecForName(flags.Env).KeyWhitelist)
	actor = loadOrCreateForest(flags, flags.ActorFile, numActions)
	critic = loadOrCreateForest(flags, flags.CriticFile, 1)
	return
}

func loadOrCreateForest(flags *Flags, path string, dims int) *treeagent.Forest {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		log.Println("Creating new forest for:", path)
		return treeagent.NewForest(dims)
	}
	var res *treeagent.Forest
	must(json.Unmarshal(data, &res))
	log.Println("Loaded forest from:", path)
	return res
}

func actionEntropy(c anyvec.Creator, r *anyrl.RolloutSet) anyvec.Numeric {
	outSeq := lazyseq.TapeRereader(c, r.AgentOuts)
	entropyer := anyrl.Softmax{}
	entropies := lazyseq.Map(outSeq, entropyer.Entropy)
	return anyvec.Sum(lazyseq.Mean(entropies).Output())
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
