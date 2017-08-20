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
	default:
		return errors.New("unknown algorithm: " + s)
	}
	return nil
}

// AddFlag adds the flag to the flag package's global set
// of flags.
func (a *AlgorithmFlag) AddFlag() {
	flag.Var(a, "algo", "splitting heuristic (sum, mse, mean, balancedsum, stddev, sign)")
}

// MuniverseEnvFlags holds various parameters for creating
// MuniverseEnvs with NewMuniverseEnvs.
type MuniverseEnvFlags struct {
	// Name is the name of the muniverse environment.
	Name string

	// RecordDir is an optional path to where recordings
	// should be stored.
	RecordDir string

	// FrameTime is the time per step.
	FrameTime time.Duration
}

// AddFlags adds the options to the flag package's global
// set of flags.
func (m *MuniverseEnvFlags) AddFlags() {
	flag.StringVar(&m.Name, "env", "", "muniverse environment name")
	flag.StringVar(&m.RecordDir, "record", "", "muniverse recording directory")
	flag.DurationVar(&m.FrameTime, "frametime", time.Second/8, "simulated time per step")
}
