package treeagent

import "github.com/unixpickle/weakai/idtrees"

// A Classifier is a decision tree or random forest.
//
// Attributes in the sample are integers corresponding to
// feature indices.
// Attribute values are float64 values corresponding to
// numerical feature values.
//
// Classes is the result are integers corresponding to
// action indices.
type Classifier interface {
	Classify(sample idtrees.AttrMap) map[idtrees.Class]float64
}

// A UniformClassifier is a Classifier which outputs an
// equal probability for every class.
type UniformClassifier struct {
	NumClasses int
}

// Classify returns a uniform distribution.
func (u *UniformClassifier) Classify(s idtrees.AttrMap) map[idtrees.Class]float64 {
	res := map[idtrees.Class]float64{}
	for i := 0; i < u.NumClasses; i++ {
		res[i] = 1 / float64(u.NumClasses)
	}
	return res
}

// A Policy is an RL policy based on a Classifier.
type Policy struct {
	// Classifier converts feature maps into action
	// probabilities.
	Classifier Classifier

	// NumActions is the number of discrete actions.
	NumActions int

	// Epsilon is the probability of taking a random
	// action rather than an action according to the
	// policy distribution.
	//
	// Set this to a small, non-zero value.
	Epsilon float64
}

// Classify applies the policy and returns a distribution
// over the classes.
func (p *Policy) Classify(sample idtrees.AttrMap) map[idtrees.Class]float64 {
	baseClassification := p.Classifier.Classify(sample)
	res := map[idtrees.Class]float64{}

	// Apply a bit of uniformity to the distribution.
	softener := p.Epsilon / (1 - p.Epsilon)
	perBin := softener / float64(p.NumActions)
	normalizer := 1 / (1 + softener)
	for i := 0; i < p.NumActions; i++ {
		res[i] = baseClassification[i] + perBin
		res[i] *= normalizer
	}

	return res
}
