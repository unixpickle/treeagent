package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"math"
	"os"
	"runtime"
	"sync"

	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/treeagent"
	"github.com/unixpickle/treeagent/experiments"
)

type Flags struct {
	EnvFlags  experiments.EnvFlags
	Algorithm experiments.AlgorithmFlag

	BatchSize    int
	ParallelEnvs int
	Depth        int
	MinLeaf      int
	StepSize     float64
	Discount     float64
	EntropyReg   float64
	SignOnly     bool
	SaveFile     string
}

func main() {
	flags := &Flags{}
	flags.EnvFlags.AddFlags()
	flags.Algorithm.AddFlag()
	flag.IntVar(&flags.BatchSize, "batch", 2048, "steps per batch")
	flag.IntVar(&flags.ParallelEnvs, "numparallel", runtime.GOMAXPROCS(0),
		"parallel environments")
	flag.IntVar(&flags.Depth, "depth", 3, "tree depth")
	flag.IntVar(&flags.MinLeaf, "minleaf", 1, "minimum samples per leaf")
	flag.Float64Var(&flags.StepSize, "step", 0.8, "step size")
	flag.Float64Var(&flags.Discount, "discount", 0, "discount factor (0 is no discount)")
	flag.Float64Var(&flags.EntropyReg, "reg", 0.01, "entropy regularization coefficient")
	flag.BoolVar(&flags.SignOnly, "sign", false, "only use sign from trees")
	flag.StringVar(&flags.SaveFile, "out", "policy.json", "file for saved policy")
	flag.Parse()
	log.Println("Run with arguments:", os.Args[1:])

	creator := anyvec32.CurrentCreator()

	log.Println("Creating environments...")
	envs, err := experiments.MakeEnvs(&flags.EnvFlags, flags.ParallelEnvs)
	must(err)
	info, _ := experiments.LookupEnvInfo(flags.EnvFlags.Name)

	var judger anypg.ActionJudger
	if flags.Discount != 0 {
		judger = &anypg.QJudger{Discount: flags.Discount, Normalize: true}
	} else {
		judger = &anypg.TotalJudger{Normalize: true}
	}

	roller := experiments.EnvRoller(creator, info, loadOrCreatePolicy(flags))

	pg := &treeagent.PG{
		Builder: treeagent.Builder{
			MaxDepth:  flags.Depth,
			Algorithm: flags.Algorithm.Algorithm,
			MinLeaf:   flags.MinLeaf,
		},
		ActionSpace: info.ActionSpace,
		Regularizer: &anypg.EntropyReg{
			Entropyer: info.ActionSpace,
			Coeff:     flags.EntropyReg,
		},
	}

	// Train on a background goroutine so that we can
	// listen for Ctrl+C on the main goroutine.
	var trainLock sync.Mutex
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			rollouts, entropy, err := experiments.GatherRollouts(roller, envs,
				flags.BatchSize)
			must(err)

			log.Printf(
				"batch %d: mean=%f stddev=%f entropy=%f count=%d",
				batchIdx,
				rollouts.Rewards.Mean(), math.Sqrt(rollouts.Rewards.Variance()),
				entropy,
				len(rollouts.Rewards),
			)

			log.Println("Training on batch...")
			advantages := judger.JudgeActions(rollouts)
			sampleChan := treeagent.RolloutSamples(rollouts, advantages)
			sampleChan = experiments.EnvSamples(info, sampleChan)
			tree, _, _ := pg.Build(treeagent.AllSamples(sampleChan))
			if flags.SignOnly {
				tree = treeagent.SignTree(tree)
			}
			roller.Policy.Add(tree, flags.StepSize)

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

func loadOrCreatePolicy(flags *Flags) *treeagent.Forest {
	data, err := ioutil.ReadFile(flags.SaveFile)
	if err != nil {
		log.Println("Created new policy.")
		info, _ := experiments.LookupEnvInfo(flags.EnvFlags.Name)
		return treeagent.NewForest(info.ParamSize)
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
