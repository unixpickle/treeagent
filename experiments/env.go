package experiments

import (
	"io"

	"github.com/unixpickle/anyrl"
)

// ActionSpace is used to parameterize actions for an
// environment.
type ActionSpace interface {
	anyrl.LogProber
	anyrl.Sampler
	anyrl.Entropyer
}

// Env is an environment with a Close() method for
// releasing the environment's resources.
type Env interface {
	io.Closer
	anyrl.Env
}

// CloseEnvs closes every environment in the list.
func CloseEnvs(envs []Env) {
	for _, e := range envs {
		e.Close()
	}
}
