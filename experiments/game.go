package experiments

import (
	"errors"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
)

// ActionSpace is used to parameterize actions for an
// environment.
type ActionSpace interface {
	anyrl.LogProber
	anyrl.Sampler
	anyrl.Entropyer
}

// GameInfo stores information about a pixel-based game.
type GameInfo struct {
	// Name of the environment.
	Name string

	// Information about actions and their parameters.
	ActionSpace ActionSpace
	ParamSize   int

	// Screen information (may be subsampled).
	Width       int
	Height      int
	NumFeatures int

	// The collection to which the game belongs.
	Muniverse bool
	Atari     bool
}

// LookupGameInfo finds information about a game based on
// the environment name.
func LookupGameInfo(name string) (*GameInfo, error) {
	spec := muniverse.SpecForName(name)
	if spec != nil {
		w, h := downsampledSize(spec.Width, spec.Height)
		// TODO: support tap games with Bernoulli actions.
		return &GameInfo{
			Name:        name,
			ActionSpace: anyrl.Softmax{},
			ParamSize:   len(spec.KeyWhitelist) + 1,
			Width:       w,
			Height:      h,
			NumFeatures: w * h,
			Muniverse:   true,
		}, nil
	}

	// TODO: add all games here.
	atariGames := map[string]bool{"Pong-v0": true}
	if atariGames[name] {
		return &GameInfo{
			Name:        name,
			ActionSpace: anyrl.Softmax{},
			ParamSize:   6,
			Width:       80,
			Height:      105,
			NumFeatures: 80 * 105,
			Atari:       true,
		}, nil
	}

	return nil, errors.New("lookup game environment: \"" + name + "\" not found")
}

// MakeGames creates n instances of a game environment.
func MakeGames(c anyvec.Creator, g *GameFlags, n int) (envs []anyrl.Env, err error) {
	defer essentials.AddCtxTo("make games ("+g.Name+")", &err)
	info, err := LookupGameInfo(g.Name)
	if err != nil {
		return nil, err
	}
	if info.Muniverse {
		return newMuniverseEnvs(c, g, n)
	} else if info.Atari {
		// TODO: atari support.
		return nil, errors.New("Atari NYI")
	} else {
		return nil, errors.New("unknown game source")
	}
}

func downsampledSize(width, height int) (int, int) {
	subWidth := width / 4
	subHeight := height / 4
	if width%4 != 0 {
		subWidth++
	}
	if height%4 != 0 {
		subHeight++
	}
	return subWidth, subHeight
}
