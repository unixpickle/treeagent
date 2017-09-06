package experiments

import (
	"math"

	"github.com/unixpickle/anyrl"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
)

var mujocoActionSizes = map[string]int{
	"Reacher-v1":                2,
	"HalfCheetah-v1":            6,
	"InvertedDoublePendulum-v1": 1,
	"InvertedPendulum-v1":       1,
	"Swimmer-v1":                2,
	"Walker2d-v1":               6,
}

var mujocoObservationSizes = map[string]int{
	"Reacher-v1":                11,
	"HalfCheetah-v1":            17,
	"InvertedDoublePendulum-v1": 11,
	"InvertedPendulum-v1":       4,
	"Swimmer-v1":                8,
	"Walker2d-v1":               17,
}

func mujocoEnvInfo(name string) (numActions, numObs int, ok bool) {
	numActions, _ = mujocoActionSizes[name]
	numObs, ok = mujocoObservationSizes[name]
	return
}

type mujocoEnv struct {
	anyrl.Env
	Closer gym.Env
	Min    []float64
	Max    []float64
}

func newMuJoCoEnvs(e *EnvFlags, n int) ([]Env, error) {
	var res []Env
	for i := 0; i < n; i++ {
		client, env, err := createGymEnv(e)
		if err != nil {
			CloseEnvs(res)
			return nil, err
		}
		actSpace, err := client.ActionSpace()
		if err != nil {
			CloseEnvs(res)
			return nil, err
		}
		var realEnv Env = &mujocoEnv{
			Env:    env,
			Closer: client,
			Min:    actSpace.Low,
			Max:    actSpace.High,
		}
		if e.History {
			realEnv = &historyEnv{Env: realEnv}
		}
		res = append(res, realEnv)
	}
	return res, nil
}

func (m *mujocoEnv) Step(action []float64) (obs []float64, reward float64,
	done bool, err error) {
	scaled := make([]float64, len(action))
	for i, x := range action {
		clamped := (math.Max(math.Min(x, 1), -1) + 1) / 2
		scaled[i] = m.Min[i] + (m.Max[i]-m.Min[i])*clamped
	}
	return m.Env.Step(scaled)
}

func (m *mujocoEnv) Close() error {
	return m.Closer.Close()
}
