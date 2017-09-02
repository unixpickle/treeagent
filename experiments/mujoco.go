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
	obs, reward, done, err = m.Env.Step(m.scaledAction(action))
	if err != nil {
		return
	}
	return
}

func (m *mujocoEnv) Close() error {
	return m.Closer.Close()
}

func (m *mujocoEnv) scaledAction(action anyvec.Vector) anyvec.Vector {
	// Make sure the actions are in a reasonable range.
	//
	// See, for example,
	// https://github.com/openai/baselines/blob/902ffcb7674dd9f3c08a0037ae57ada852f13d74/baselines/acktr/acktr_cont.py#L36

	c := action.Creator()

	res := action.Copy()
	res.AddScalar(c.MakeNumeric(1))
	res.Scale(c.MakeNumeric(0.5))
	anyvec.ClipPos(res)
	lessMask := res.Copy()
	anyvec.LessThan(lessMask, c.MakeNumeric(1))
	res.Mul(lessMask)

	diff := m.Max.Copy()
	diff.Sub(m.Min)
	res.Mul(diff)
	res.Add(m.Min)

	return res
}
