package treeagent

// SignTree copies the tree and sets every parameter to 1
// or -1 depending on its sign.
func SignTree(t *Tree) *Tree {
	if t.Leaf {
		return &Tree{
			Leaf:   true,
			Params: ActionParams(smallVec(t.Params).Copy().Signs()),
		}
	}
	return &Tree{
		Feature:      t.Feature,
		Threshold:    t.Threshold,
		LessThan:     SignTree(t.LessThan),
		GreaterEqual: SignTree(t.GreaterEqual),
	}
}
