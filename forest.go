package treeagent

// ActionDist is a probability distribution over actions.
// It maps each action index to a probability.
type ActionDist []float64

// Forest is a decision forest.
//
// The Trees in a Forest are weighted geometrically.
// Each time a tree is added, the forest output becomes:
//
//     (1-step)*oldForest + step*newTree
//
// Before any trees are added, the forest uses Base as the
// action distribution.
type Forest struct {
	Step  float64
	Base  ActionDist
	Trees []*Tree
}

// NewForest creates an empty forest with a uniform
// distribution and the given step size.
func NewForest(step float64, numActions int) *Forest {
	base := make(ActionDist, numActions)
	for i := range base {
		base[i] = 1 / float64(numActions)
	}
	return &Forest{
		Step: step,
		Base: base,
	}
}

// Add adds a tree to the forest.
func (f *Forest) Add(t *Tree) {
	f.Trees = append(f.Trees, t)
}

// Truncate removes old trees so that the forest only
// contains n trees.
func (f *Forest) Truncate(n int) {
	if n >= len(f.Trees) {
		return
	}
	f.Trees = append([]*Tree{}, f.Trees[len(f.Trees)-n:]...)
}

// Apply runs the features through each tree in the forest
// and produces an aggregate action distribution.
func (f *Forest) Apply(features []float64) ActionDist {
	dist := append(ActionDist{}, f.Base...)
	for _, tree := range f.Trees {
		for i := range dist {
			dist[i] *= 1 - f.Step
		}
		dist[tree.Find(features)] += f.Step
	}
	return dist
}

// Tree is a node in a decision tree.
//
// A Tree is either a leaf node or a branching node.
type Tree struct {
	Leaf bool

	// Information for leaf nodes.
	Action int

	// Information for branching nodes.
	Feature      int
	Threshold    float64
	LessThan     *Tree
	GreaterEqual *Tree
}

// Find returns the action for the feature vector.
func (t *Tree) Find(features []float64) int {
	if t.Leaf {
		return t.Action
	}
	val := features[t.Feature]
	if val < t.Threshold {
		return t.LessThan.Find(features)
	} else {
		return t.GreaterEqual.Find(features)
	}
}
