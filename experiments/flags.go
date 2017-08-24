package experiments

import (
	"errors"
	"flag"
	"strings"
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
	return a.Algorithm.String()
}

// Set sets the algorithm from a string representation.
func (a *AlgorithmFlag) Set(s string) error {
	for _, alg := range treeagent.TreeAlgorithms {
		if alg.String() == s {
			a.Algorithm = alg
			return nil
		}
	}
	return errors.New("unknown algorithm: " + s)
}

// AddFlag adds the flag to the flag package's global set
// of flags.
func (a *AlgorithmFlag) AddFlag() {
	var names []string
	for _, alg := range treeagent.TreeAlgorithms {
		names = append(names, alg.String())
	}
	flag.Var(a, "algo", "splitting heuristic ("+strings.Join(names, ", ")+")")
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

	// History, if true, indicates that the previous
	// observation should be concatenated with the current
	// one to form a bigger observation.
	History bool
}

// AddFlags adds the options to the flag package's global
// set of flags.
func (g *GameFlags) AddFlags() {
	flag.StringVar(&g.Name, "env", "", "game environment name")
	flag.StringVar(&g.RecordDir, "record", "", "muniverse recording directory")
	flag.DurationVar(&g.FrameTime, "frametime", time.Second/8,
		"time per step (muniverse only)")
	flag.StringVar(&g.GymHost, "gym", "localhost:5001", "host for gym-socket-api")
	flag.BoolVar(&g.History, "history", false, "use both current and last observation")
}
