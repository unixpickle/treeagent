package treeagent

import "math"

// ActionDist is a probability distribution over actions.
// It maps each action (as identified by an index) to the
// action's log probability.
type ActionDist []float64

// NewActionDist creates a uniform distribution.
func NewActionDist(numActions int) ActionDist {
	res := make(ActionDist, numActions)
	for i := range res {
		res[i] = math.Log(1.0 / float64(numActions))
	}
	return res
}

func zeroActionDist(length int) ActionDist {
	res := make(ActionDist, length)
	for i := range res {
		res[i] = math.Inf(-1)
	}
	return res
}

func (a ActionDist) copy() ActionDist {
	return append(ActionDist{}, a...)
}

func (a ActionDist) normalize() ActionDist {
	sum := math.Inf(-1)
	for _, v := range a {
		sum = addLogs(sum, v)
	}
	res := make(ActionDist, len(a))
	for k, v := range a {
		res[k] = v - sum
	}
	return res
}

func (a ActionDist) exp() ActionDist {
	res := make(ActionDist, len(a))
	for k, v := range a {
		res[k] = math.Exp(v)
	}
	return res
}

func (a ActionDist) dot(a1 ActionDist) float64 {
	var res float64
	for action, p := range a {
		p1 := a1[action]

		// Deal with 0*log(0).
		if p == 0 && math.IsInf(p1, -1) ||
			p1 == 0 && math.IsInf(p, -1) {
			continue
		}

		res += p * p1
	}
	return res
}

func (a ActionDist) add(a1 ActionDist) ActionDist {
	res := make([]float64, len(a))
	for action, p := range a {
		res[action] = addLogs(p, a1[action])
	}
	return res
}

func addActionDists(dists []ActionDist) ActionDist {
	sum := dists[0]
	for _, dist := range dists[1:] {
		sum = sum.add(dist)
	}
	return sum
}

func addLogs(l1, l2 float64) float64 {
	if math.IsInf(l1, -1) {
		return l2
	} else if math.IsInf(l2, -1) {
		return l1
	}
	max := math.Max(l1, l2)
	return max + math.Log(math.Exp(l1-max)+math.Exp(l2-max))
}
