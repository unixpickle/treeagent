package main

import (
	"compress/flate"
	"flag"
	"fmt"
	"log"
	"math"
	"runtime"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/treeagent"
	"github.com/unixpickle/treeagent/experiments"
)

type Flags struct {
	GameFlags experiments.GameFlags

	Discount     float64
	Unnormalized bool

	NumParallel int
	Batch       int

	Depth     int
	MinLeaf   int
	MaskParam int
	ValueFunc bool

	DumpLeaves bool
}

func main() {
	var flags Flags
	flags.GameFlags.AddFlags()
	flag.Float64Var(&flags.Discount, "discount", 0.7, "reward discount factor")
	flag.BoolVar(&flags.Unnormalized, "unnorm", false, "use unnormalized rewards")
	flag.IntVar(&flags.NumParallel, "numparallel", runtime.GOMAXPROCS(0),
		"environments to run in parallel")
	flag.IntVar(&flags.Batch, "batch", 2048, "number of steps to gather")
	flag.IntVar(&flags.Depth, "depth", 4, "depth of trees")
	flag.IntVar(&flags.MinLeaf, "minleaf", 1, "minimum samples per leaf")
	flag.IntVar(&flags.MaskParam, "mask", -1, "specific parameter to fit")
	flag.BoolVar(&flags.ValueFunc, "valfunc", false, "train a value function, not a policy")
	flag.BoolVar(&flags.DumpLeaves, "dump", false, "print all leaves")
	flag.Parse()

	log.Println("Creating environments...")
	c := anyvec32.CurrentCreator()
	envs, err := experiments.MakeGames(c, &flags.GameFlags, flags.NumParallel)
	essentials.Must(err)
	info, _ := experiments.LookupGameInfo(flags.GameFlags.Name)

	log.Println("Gathering rollouts...")
	roller := &treeagent.Roller{
		Policy:      treeagent.NewForest(info.ParamSize),
		Creator:     c,
		ActionSpace: info.ActionSpace,
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}
	rollouts, _, err := experiments.GatherRollouts(roller, envs, flags.Batch)
	experiments.CloseEnvs(envs)
	essentials.Must(err)

	log.Println("Creating training samples...")
	judger := &anypg.QJudger{
		Discount:  flags.Discount,
		Normalize: !flags.Unnormalized,
	}
	advs := judger.JudgeActions(rollouts)
	rawSamples := treeagent.RolloutSamples(rollouts, advs)
	samples := treeagent.AllSamples(treeagent.Uint8Samples(rawSamples))

	log.Println("Computing exact gradient...")
	exactGrad := ExactGradient(samples, info.ActionSpace)

	log.Println("Building trees...")

	algos := treeagent.TreeAlgorithms
	if flags.ValueFunc {
		algos = []treeagent.TreeAlgorithm{treeagent.MSEAlgorithm}
	}
	for _, algo := range algos {
		PrintSeparator()
		name := (&experiments.AlgorithmFlag{Algorithm: algo}).String()
		fmt.Println("Algorithm:", name)

		var tree *treeagent.Tree
		if flags.ValueFunc {
			judger := &treeagent.Judger{
				ValueFunc: treeagent.NewForest(1),
				Discount:  flags.Discount,
			}
			tree, _ = judger.Train(samples, flags.Depth)
		} else {
			builder := &treeagent.Builder{
				MaxDepth:    flags.Depth,
				ActionSpace: info.ActionSpace,
				Algorithm:   algo,
				MinLeaf:     flags.MinLeaf,
			}
			if flags.MaskParam >= 0 {
				builder.ParamWhitelist = []int{flags.MaskParam}
			}
			tree = builder.Build(samples)
		}
		TreeAnalysis(tree, samples, &flags)
		if !flags.ValueFunc {
			GradAnalysis(tree, samples, exactGrad)
		}
	}
	PrintSeparator()
}

func TreeAnalysis(tree *treeagent.Tree, samples []treeagent.Sample, flags *Flags) {
	visitation := Visitation(tree, samples)
	if flags.DumpLeaves {
		DumpLeaves(tree, visitation)
	}

	meanCount, stddevCount := VisitationStats(visitation)
	fmt.Println("Visitation mean:", meanCount)
	fmt.Println("Visitation stddev:", stddevCount)

	meanParam, stddevParam := ParamStats(visitation)
	fmt.Println("Param mean:", meanParam)
	fmt.Println("Param stddev:", stddevParam)
}

func DumpLeaves(tree *treeagent.Tree, counts map[*treeagent.Tree]int) {
	if tree.Leaf {
		fmt.Printf("%v (%d samples)\n", tree.Params, counts[tree])
	} else {
		DumpLeaves(tree.LessThan, counts)
		DumpLeaves(tree.GreaterEqual, counts)
	}
}

func Visitation(t *treeagent.Tree, s []treeagent.Sample) map[*treeagent.Tree]int {
	res := map[*treeagent.Tree]int{}
	var visit func(sample treeagent.Sample, t *treeagent.Tree)
	visit = func(sample treeagent.Sample, t *treeagent.Tree) {
		res[t]++
		if !t.Leaf {
			if sample.Feature(t.Feature) < t.Threshold {
				visit(sample, t.LessThan)
			} else {
				visit(sample, t.GreaterEqual)
			}
		}
	}
	for _, sample := range s {
		visit(sample, t)
	}
	return res
}

// VisitationStats returns the mean and standard deviation
// for leaf visitations.
func VisitationStats(visitation map[*treeagent.Tree]int) (mean, stddev float64) {
	var counts []float64
	for node, count := range visitation {
		if node.Leaf {
			counts = append(counts, float64(count))
		}
	}
	means, stddevs := Stats(counts)
	return means[0], stddevs[0]
}

// ParamStats returns the means and standard deviations
// for each parameter.
func ParamStats(visitation map[*treeagent.Tree]int) (mean, stddev []float64) {
	var paramVals [][]float64
	for node := range visitation {
		if node.Leaf {
			if paramVals == nil {
				paramVals = make([][]float64, len(node.Params))
			}
			for i, p := range node.Params {
				paramVals[i] = append(paramVals[i], p)
			}
		}
	}
	return Stats(paramVals...)
}

// Stats computes means and standard deviations.
func Stats(lists ...[]float64) (means, stddevs []float64) {
	for _, list := range lists {
		var sum float64
		var sqSum float64
		for _, x := range list {
			sum += x
			sqSum += x * x
		}
		mean := sum / float64(len(list))
		secondMoment := sqSum / float64(len(list))
		means = append(means, mean)
		stddevs = append(stddevs, math.Sqrt(secondMoment-mean*mean))
	}
	return
}

func PrintSeparator() {
	fmt.Println("--------------------------")
}
