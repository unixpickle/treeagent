package treeagent

// Tree is a node in a decision tree.
//
// A Tree is either a leaf node or a branching node.
// Leaf nodes have a non-nil Distribution for deciding on
// actions.
// Branching nodes provide two children and a feature to
// decide which branch to take.
type Tree struct {
	// Distribution for leaf node.
	Distribution ActionDist

	// Information for branching nodes.
	Feature      int
	Threshold    float64
	LessThan     *Tree
	GreaterEqual *Tree
}

// Find returns the leaf node for the feature vector.
func (t *Tree) Find(features []float64) ActionDist {
	if t.Distribution != nil {
		return t.Distribution
	}
	val := features[t.Feature]
	if val < t.Threshold {
		return t.LessThan.Find(features)
	} else {
		return t.GreaterEqual.Find(features)
	}
}
