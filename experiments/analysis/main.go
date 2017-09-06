package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"runtime"

	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/treeagent"
	"github.com/unixpickle/treeagent/experiments"
)

type Flags struct {
	EnvFlags experiments.EnvFlags

	Discount     float64
	Unnormalized bool

	NumParallel int
	Batch       int
	ValBatch    int

	Depth       int
	MinLeaf     int
	MinLeafFrac float64
	MaskParam   int
	ValueFunc   bool

	DumpLeaves bool
}

func main() {
	var flags Flags
	flags.EnvFlags.AddFlags()
	flag.Float64Var(&flags.Discount, "discount", 0.7, "reward discount factor")
	flag.BoolVar(&flags.Unnormalized, "unnorm", false, "use unnormalized rewards")
	flag.IntVar(&flags.NumParallel, "numparallel", runtime.GOMAXPROCS(0),
		"environments to run in parallel")
	flag.IntVar(&flags.Batch, "batch", 2048, "number of steps to gather")
	flag.IntVar(&flags.ValBatch, "valbatch", 0, "number of validation steps to gather")
	flag.IntVar(&flags.Depth, "depth", 4, "depth of trees")
	flag.IntVar(&flags.MinLeaf, "minleaf", 1, "minimum samples per leaf")
	flag.Float64Var(&flags.MinLeafFrac, "minleaffrac", 0,
		"minimum fraction of samples per leaf")
	flag.IntVar(&flags.MaskParam, "mask", -1, "specific parameter to fit")
	flag.BoolVar(&flags.ValueFunc, "valfunc", false, "train a value function, not a policy")
	flag.BoolVar(&flags.DumpLeaves, "dump", false, "print all leaves")
	flag.Parse()

	c := anyvec32.CurrentCreator()
	info, err := experiments.LookupEnvInfo(flags.EnvFlags.Name)
	essentials.Must(err)

	log.Println("Creating training samples...")
	samples := GatherSamples(c, &flags, flags.Batch)

	var valSamples []treeagent.Sample
	if flags.ValBatch > 0 {
		log.Println("Creating validation samples...")
		valSamples = GatherSamples(c, &flags, flags.ValBatch)
	}

	log.Println("Computing exact gradient...")
	exactGrad := ExactGradient(samples, info.ActionSpace)

	var valGrad anyvec.Vector
	if valSamples != nil {
		valGrad = ExactGradient(valSamples, info.ActionSpace)
	}

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
				MaxDepth:    flags.Depth,
				ValueFunc:   treeagent.NewForest(1),
				Discount:    flags.Discount,
				MinLeaf:     flags.MinLeaf,
				MinLeafFrac: flags.MinLeafFrac,
			}
			tree, _ = judger.Train(samples)
		} else {
			pg := &treeagent.PG{
				Builder: treeagent.Builder{
					MaxDepth:    flags.Depth,
					Algorithm:   algo,
					MinLeaf:     flags.MinLeaf,
					MinLeafFrac: flags.MinLeafFrac,
				},
				ActionSpace: info.ActionSpace,
			}
			if flags.MaskParam >= 0 {
				pg.Builder.ParamWhitelist = []int{flags.MaskParam}
			}
			tree, _, _ = pg.Build(samples)
		}
		TreeAnalysis(tree, samples, &flags)
		if !flags.ValueFunc {
			GradAnalysis("Training", tree, samples, exactGrad)
			if valGrad != nil {
				GradAnalysis("Validation", tree, valSamples, valGrad)
			}
		}
	}
	PrintSeparator()
}

func GatherSamples(c anyvec.Creator, flags *Flags, numSteps int) []treeagent.Sample {
	envs, err := experiments.MakeEnvs(&flags.EnvFlags, flags.NumParallel)
	essentials.Must(err)
	defer experiments.CloseEnvs(envs)
	info, _ := experiments.LookupEnvInfo(flags.EnvFlags.Name)

	roller := experiments.EnvRoller(c, info, treeagent.NewForest(info.ParamSize))
	rollouts, _, err := experiments.GatherRollouts(roller, envs, flags.Batch)
	essentials.Must(err)

	judger := &anypg.QJudger{
		Discount:  flags.Discount,
		Normalize: !flags.Unnormalized,
	}
	advs := judger.JudgeActions(rollouts)

	sampleChan := treeagent.RolloutSamples(rollouts, advs)
	sampleChan = experiments.EnvSamples(info, sampleChan)
	return treeagent.AllSamples(sampleChan)
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
