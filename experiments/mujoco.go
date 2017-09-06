package experiments

import (
	"math"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
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
	Min    anyvec.Vector
	Max    anyvec.Vector
}

func newMuJoCoEnvs(c anyvec.Creator, e *EnvFlags, n int) ([]Env, error) {
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
			Min:    c.MakeVectorData(c.MakeNumericList(actSpace.Low)),
			Max:    c.MakeVectorData(c.MakeNumericList(actSpace.High)),
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
	clamped := make([]float64, len(action))
	for i, x := range action {
		clamped[i] = math.Max(math.Min(x, 1), -1)
	}
	return m.Env.Step(clamped)
}

func (m *mujocoEnv) Close() error {
	return m.Closer.Close()
}
