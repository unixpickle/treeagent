package treeagent

// ActionParams is a probability distribution represented
// as parameters for an action distribution.
type ActionParams []float64

// A Forest is a weighted ensemble of trees.
//
// Before any trees are added, the Forest uses Base as the
// parameter vector.
// As trees are added, their weighted outputs are added to
// the parameters in Base.
type Forest struct {
	Base    ActionParams
	Trees   []*Tree
	Weights []float64
}

// NewForest creates an empty forest with a set of zero
// parameters.
func NewForest(paramDim int) *Forest {
	return &Forest{Base: make(ActionParams, paramDim)}
}

// Add adds a tree to the forest.
func (f *Forest) Add(tree *Tree, weight float64) {
	f.Trees = append(f.Trees, tree)
	f.Weights = append(f.Weights, weight)
}

// Apply runs the features through each Tree and produces
// a parameter vector.
func (f *Forest) Apply(features []float64) ActionParams {
	params := append(ActionParams{}, f.Base...)
	for i, tree := range f.Trees {
		w := f.Weights[i]
		for j, x := range tree.Find(features) {
			params[j] += x * w
		}
	}
	return params
}

// Tree is a node in a decision tree.
//
// A Tree is either a leaf node or a branching node.
type Tree struct {
	Leaf bool

	// Information for leaf nodes.
	Params ActionParams

	// Information for branching nodes.
	Feature      int
	Threshold    float64
	LessThan     *Tree
	GreaterEqual *Tree
}

// Find finds the leaf parameters for the features.
func (t *Tree) Find(features []float64) ActionParams {
	if t.Leaf {
		return t.Params
	}
	val := features[t.Feature]
	if val < t.Threshold {
		return t.LessThan.Find(features)
	} else {
		return t.GreaterEqual.Find(features)
	}
}

func (t *Tree) scaleParams(scale float64) {
	if t.Leaf {
		for i, x := range t.Params {
			t.Params[i] = x * scale
		}
	} else {
		t.LessThan.scaleParams(scale)
		t.GreaterEqual.scaleParams(scale)
	}
}
