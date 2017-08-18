package treeagent

// SignTree copies the tree and sets every parameter to 1
// or -1 depending on its sign.
func SignTree(t *Tree) *Tree {
	if t.Leaf {
		res := &Tree{Leaf: true, Params: make(ActionParams, len(t.Params))}
		for i, p := range t.Params {
			if p > 0 {
				res.Params[i] = 1
			} else if p < 0 {
				res.Params[i] = -1
			}
		}
		return res
	}
	return &Tree{
		Feature:      t.Feature,
		Threshold:    t.Threshold,
		LessThan:     SignTree(t.LessThan),
		GreaterEqual: SignTree(t.GreaterEqual),
	}
}
