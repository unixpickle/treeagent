package experiments

import (
	"errors"
	"flag"
	"time"

	"github.com/unixpickle/treeagent"
)

// AlgorithmFlag is a flag.Value for a treeagent
// algorithm.
type AlgorithmFlag struct {
	Algorithm treeagent.TreeAlgorithm
}

// String returns the string representation of the
// algorithm.
func (a *AlgorithmFlag) String() string {
	switch a.Algorithm {
	case treeagent.SumAlgorithm:
		return "sum"
	case treeagent.MeanAlgorithm:
		return "mean"
	case treeagent.MSEAlgorithm:
		return "mse"
	case treeagent.BalancedSumAlgorithm:
		return "balancedsum"
	case treeagent.StddevAlgorithm:
		return "stddev"
	case treeagent.SignAlgorithm:
		return "sign"
	case treeagent.AbsAlgorithm:
		return "abs"
	default:
		return ""
	}
}

// Set sets the algorithm from a string representation.
func (a *AlgorithmFlag) Set(s string) error {
	switch s {
	case "sum":
		a.Algorithm = treeagent.SumAlgorithm
	case "mean":
		a.Algorithm = treeagent.MeanAlgorithm
	case "mse":
		a.Algorithm = treeagent.MSEAlgorithm
	case "balancedsum":
		a.Algorithm = treeagent.BalancedSumAlgorithm
	case "stddev":
		a.Algorithm = treeagent.StddevAlgorithm
	case "sign":
		a.Algorithm = treeagent.SignAlgorithm
	case "abs":
		a.Algorithm = treeagent.AbsAlgorithm
	default:
		return errors.New("unknown algorithm: " + s)
	}
	return nil
}

// AddFlag adds the flag to the flag package's global set
// of flags.
func (a *AlgorithmFlag) AddFlag() {
	flag.Var(a, "algo", "splitting heuristic (sum, mse, mean, balancedsum, "+
		"stddev, sign, abs)")
}

// GameFlags holds various parameters for creating game
// environments.
type GameFlags struct {
	// Name is the name of the environment.
	Name string

	// RecordDir is an optional path to where recordings
	// should be stored.
	// Currently, this is only supported for muniverse.
	RecordDir string

	// FrameTime is the time per step.
	// This is only supported for muniverse games.
	FrameTime time.Duration

	// GymHost is the destination host for an instance of
	// gym-socket-api.
	GymHost string
}

// AddFlags adds the options to the flag package's global
// set of flags.
func (g *GameFlags) AddFlags() {
	flag.StringVar(&g.Name, "env", "", "game environment name")
	flag.StringVar(&g.RecordDir, "record", "", "muniverse recording directory")
	flag.DurationVar(&g.FrameTime, "frametime", time.Second/8,
		"time per step (muniverse only)")
	flag.StringVar(&g.GymHost, "gym", "localhost:5001", "host for gym-socket-api")
}
