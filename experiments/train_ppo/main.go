package main

import (
	"compress/flate"
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/treeagent"
	"github.com/unixpickle/treeagent/experiments"
)

type Flags struct {
	EnvFlags  experiments.EnvFlags
	Algorithm experiments.AlgorithmFlag

	BatchSize    int
	ParallelEnvs int

	Depth       int
	MinLeaf     int
	TreeDecay   float64
	MaxTrees    int
	StepSize    float64
	ValStep     float64
	TuneStep    float64
	Discount    float64
	Lambda      float64
	FeatureFrac float64
	Minibatch   float64
	EntropyReg  float64
	Epsilon     float64
	SignOnly    bool
	Iters       int
	ValIters    int
	TuneIters   int
	CoordDesc   bool

	ActorFile  string
	CriticFile string
}

func main() {
	flags := &Flags{}
	flags.EnvFlags.AddFlags()
	flags.Algorithm.AddFlag()
	flag.IntVar(&flags.BatchSize, "batch", 2048, "steps per rollout")
	flag.IntVar(&flags.ParallelEnvs, "numparallel", runtime.GOMAXPROCS(0),
		"parallel environments")
	flag.IntVar(&flags.Depth, "depth", 8, "tree depth")
	flag.IntVar(&flags.MinLeaf, "minleaf", 1, "minimum samples per leaf")
	flag.Float64Var(&flags.TreeDecay, "decay", 1, "tree decay rate for value function")
	flag.IntVar(&flags.MaxTrees, "maxtrees", -1, "max trees in value function")
	flag.Float64Var(&flags.StepSize, "step", 0.8, "step size")
	flag.Float64Var(&flags.ValStep, "valstep", 1, "value function step shrinkage")
	flag.Float64Var(&flags.TuneStep, "tunestep", 1, "step size for tuning")
	flag.Float64Var(&flags.Discount, "discount", 0.8, "discount factor")
	flag.Float64Var(&flags.Lambda, "lambda", 0.95, "GAE coefficient")
	flag.Float64Var(&flags.FeatureFrac, "featurefrac", 1, "fraction of features to use")
	flag.Float64Var(&flags.Minibatch, "minibatch", 1, "mini-batch fraction for each tree")
	flag.Float64Var(&flags.EntropyReg, "reg", 0.01, "entropy regularization coefficient")
	flag.Float64Var(&flags.Epsilon, "epsilon", 0.1, "PPO epsilon")
	flag.BoolVar(&flags.SignOnly, "sign", false, "only use sign from trees")
	flag.IntVar(&flags.Iters, "iters", 4, "training iterations per batch")
	flag.IntVar(&flags.ValIters, "valiters", 4, "value training iterations per batch")
	flag.IntVar(&flags.TuneIters, "tuneiters", 0, "tuning iterations per batch")
	flag.BoolVar(&flags.CoordDesc, "coorddesc", false, "tune one action parameter at a time")
	flag.StringVar(&flags.ActorFile, "actor", "actor.json", "file for saved policy")
	flag.StringVar(&flags.CriticFile, "critic", "critic.json", "file for saved value function")
	flag.Parse()

	log.Println("Run with arguments:", os.Args[1:])

	creator := anyvec32.CurrentCreator()

	log.Println("Creating environments...")
	envs, err := experiments.MakeEnvs(creator, &flags.EnvFlags, flags.ParallelEnvs)
	must(err)
	info, _ := experiments.LookupEnvInfo(flags.EnvFlags.Name)

	policy, valueFunc := loadOrCreateForests(flags)
	roller := &treeagent.Roller{
		Policy:      policy,
		Creator:     creator,
		ActionSpace: info.ActionSpace,
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	judger := &treeagent.Judger{
		ValueFunc:   valueFunc,
		Discount:    flags.Discount,
		Lambda:      flags.Lambda,
		MaxDepth:    flags.Depth,
		FeatureFrac: flags.FeatureFrac,
		MinLeaf:     flags.MinLeaf,
	}

	ppo := &treeagent.PPO{
		PG: treeagent.PG{
			Builder: treeagent.Builder{
				MaxDepth:    flags.Depth,
				Algorithm:   flags.Algorithm.Algorithm,
				FeatureFrac: flags.FeatureFrac,
				MinLeaf:     flags.MinLeaf,
			},
			ActionSpace: info.ActionSpace,
			Regularizer: &anypg.EntropyReg{
				Entropyer: info.ActionSpace,
				Coeff:     flags.EntropyReg,
			},
		},
		Epsilon: flags.Epsilon,
	}

	var trainLock sync.Mutex
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			rollouts, entropy, err := experiments.GatherRollouts(roller, envs,
				flags.BatchSize)
			must(err)

			log.Printf(
				"batch %d: mean=%f stddev=%f entropy=%f frames=%d count=%d",
				batchIdx,
				rollouts.Rewards.Mean(), math.Sqrt(rollouts.Rewards.Variance()),
				entropy,
				rollouts.NumSteps(),
				len(rollouts.Rewards),
			)

			log.Println("Training policy...")
			advantages := judger.JudgeActions(rollouts)
			rawSamples := treeagent.RolloutSamples(rollouts, advantages)
			sampleChan := treeagent.Uint8Samples(rawSamples)
			samples := treeagent.AllSamples(sampleChan)
			for i := 0; i < flags.TuneIters; i++ {
				minibatch := treeagent.Minibatch(samples, flags.Minibatch)
				if flags.CoordDesc {
					ppo.PG.Builder.ParamWhitelist = []int{rand.Intn(info.ParamSize)}
				}
				grad, obj, reg := ppo.WeightGradient(minibatch, policy)

				// Don't take larger and larger steps as more and
				// more trees are added.
				tuneNorm := 1 / math.Sqrt(float64(len(policy.Trees)))

				policy.AddWeights(grad, flags.TuneStep*tuneNorm)
				numPruned := policy.PruneNegative()
				log.Printf("tune %d: objective=%f reg=%f prune=%d", i,
					obj, reg, numPruned)
			}
			for i := 0; i < flags.Iters; i++ {
				minibatch := treeagent.Minibatch(samples, flags.Minibatch)
				if flags.CoordDesc {
					ppo.PG.Builder.ParamWhitelist = []int{rand.Intn(info.ParamSize)}
				}
				tree, obj, reg := ppo.Build(minibatch, policy)
				log.Printf("step %d: objective=%f reg=%f", i, obj, reg)
				if flags.SignOnly {
					tree = treeagent.SignTree(tree)
				}
				policy.Add(tree, flags.StepSize)
			}

			log.Println("Training value function...")
			rawSamples = judger.TrainingSamples(rollouts)
			sampleChan = treeagent.Uint8Samples(rawSamples)
			samples = treeagent.AllSamples(sampleChan)
			for i := 0; i < flags.ValIters; i++ {
				decayForest(flags, valueFunc)
				minibatch := treeagent.Minibatch(samples, flags.Minibatch)
				tree, loss := judger.Train(minibatch)
				step := judger.OptimalWeight(samples, tree) * flags.ValStep
				valueFunc.Add(tree, step)
				log.Printf("step %d: mse=%f step=%f", i, loss, step)
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

func loadOrCreateForests(flags *Flags) (actor, critic *treeagent.Forest) {
	info, _ := experiments.LookupEnvInfo(flags.EnvFlags.Name)
	actor = loadOrCreateForest(flags, flags.ActorFile, info.ParamSize)
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

func decayForest(flags *Flags, forest *treeagent.Forest) {
	if flags.TreeDecay < 1 {
		forest.Scale(flags.TreeDecay)
	}
	if flags.MaxTrees > 0 && len(forest.Trees) >= flags.MaxTrees {
		forest.RemoveFirst()
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
