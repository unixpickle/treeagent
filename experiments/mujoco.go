package experiments

import (
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
	Env    anyrl.Env
	Closer gym.Env
	Min    anyvec.Vector
	Max    anyvec.Vector
}

func newMuJoCoEnvs(c anyvec.Creator, e *EnvFlags, n int) ([]Env, error) {
	var res []Env
	for i := 0; i < n; i++ {
		client, env, err := createGymEnv(c, e)
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

func (m *mujocoEnv) Reset() (obs anyvec.Vector, err error) {
	obs, err = m.Env.Reset()
	if err != nil {
		return
	}
	return
}

func (m *mujocoEnv) Step(action anyvec.Vector) (obs anyvec.Vector, reward float64,
	done bool, err error) {
	clamped := action.Copy()
	clampVec(clamped, m.Min, m.Max)
	obs, reward, done, err = m.Env.Step(clamped)
	if err != nil {
		return
	}
	return
}

func (m *mujocoEnv) Close() error {
	return m.Closer.Close()
}

func clampVec(vec, min, max anyvec.Vector) {
	c := vec.Creator()

	clampMin(vec, min)

	neg1 := c.MakeNumeric(-1)
	vec.Scale(neg1)
	negMax := max.Copy()
	negMax.Scale(neg1)
	clampMin(vec, negMax)
	vec.Scale(neg1)
}

func clampMin(vec, min anyvec.Vector) {
	vec.Sub(min)
	anyvec.ClipPos(vec)
	vec.Add(min)
}
