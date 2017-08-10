package treeagent

import "math"

// Action is a discrete action in an environment.
// Actions must be comparable with the == operator.
type Action interface{}

// ActionDist is a probability distribution over actions.
type ActionDist map[Action]float64

func (a ActionDist) copy() ActionDist {
	res := ActionDist{}
	for k, v := range a {
		res[k] = v
	}
	return res
}

func (a ActionDist) normalize() ActionDist {
	var sum float64
	for _, v := range a {
		sum += v
	}
	res := ActionDist{}
	for k, v := range a {
		res[k] = v / sum
	}
	return res
}

func (a ActionDist) log() ActionDist {
	res := ActionDist{}
	for k, v := range a {
		if v == 0 {
			continue
		}
		res[k] = math.Log(v)
	}
	return res
}

func (a ActionDist) sub(a1 ActionDist) ActionDist {
	res := a.copy()
	for k, v := range a1 {
		res[k] -= v
	}
	return res
}

func (a ActionDist) dot(a1 ActionDist) float64 {
	var res float64
	for k, v := range a {
		res += v * a1[k]
	}
	return res
}

func addActionDists(dists ...ActionDist) ActionDist {
	res := ActionDist{}
	for _, dist := range dists {
		for act, prob := range dist {
			res[act] += prob
		}
	}
	return res
}
