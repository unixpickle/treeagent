package experiments

import (
	"sync"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/treeagent"
)

// GatherRollouts produces a batch of rollouts by running
// the environments in parallel.
//
// The steps argument specifies the minimum number of
// timesteps in the resulting batch of rollouts.
//
// Along with the rollouts, GatherRollouts produces an
// entropy measure, indicating how much exploration took
// place.
func GatherRollouts(roller *treeagent.Roller, envs []Env,
	steps int) (*anyrl.RolloutSet, anyvec.Numeric, error) {
	resChan := make(chan *anyrl.RolloutSet, 1)
	errChan := make(chan error, 1)
	requests := make(chan struct{}, len(envs))
	for i := 0; i < len(envs); i++ {
		requests <- struct{}{}
	}

	var wg sync.WaitGroup
	for _, env := range envs {
		wg.Add(1)
		go func(env anyrl.Env) {
			defer wg.Done()
			for _ = range requests {
				rollout, err := roller.Rollout(env)
				if err != nil {
					select {
					case errChan <- err:
					default:
					}
					return
				}
				resChan <- rollout
			}
		}(env)
	}

	go func() {
		wg.Wait()
		close(resChan)
		close(errChan)
	}()

	var res []*anyrl.RolloutSet
	var totalSteps int
	for item := range resChan {
		res = append(res, item)
		if totalSteps < steps {
			totalSteps += item.NumSteps()
			if totalSteps < steps {
				requests <- struct{}{}
			} else {
				close(requests)
			}
		}
	}
	packed := anyrl.PackRolloutSets(roller.Creator(), res)

	reg := &anypg.EntropyReg{
		Entropyer: roller.ActionSpace.(anyrl.Entropyer),
		Coeff:     1,
	}
	entropy := anypg.AverageReg(roller.Creator(), packed.AgentOuts, reg)

	return packed, entropy, <-errChan
}
