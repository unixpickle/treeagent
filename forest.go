package treeagent

import (
	"runtime"
	"sync"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

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

// Scale scales all the weights by the given value.
func (f *Forest) Scale(scale float64) {
	for i := range f.Weights {
		f.Weights[i] *= scale
	}
}

// RemoveFirst removes the first tree from the forest.
func (f *Forest) RemoveFirst() {
	essentials.OrderedDelete(&f.Trees, 0)
	essentials.OrderedDelete(&f.Weights, 0)
}

// AddWeights adds a value to each tree weight.
// Weight i is updated by adding w[i]*scale.
func (f *Forest) AddWeights(w []float64, scale float64) {
	if len(w) != len(f.Weights) {
		panic("weight vectors must have the same length")
	}
	for i, x := range w {
		f.Weights[i] += x * scale
	}
}

// PruneNegative removes trees with negative or 0 weights.
// It returns the number of removed trees.
func (f *Forest) PruneNegative() int {
	var newWeights []float64
	var newTrees []*Tree
	for i, w := range f.Weights {
		if w > 0 {
			newWeights = append(newWeights, w)
			newTrees = append(newTrees, f.Trees[i])
		}
	}
	res := len(f.Trees) - len(newTrees)
	f.Weights = newWeights
	f.Trees = newTrees
	return res
}

// Apply runs the features through each Tree and produces
// a parameter vector.
func (f *Forest) Apply(features []float64) ActionParams {
	return f.ApplyFeatureSource(sliceFeatureSource(features))
}

// ApplyFeatureSource is like Apply, but for a
// FeatureSource.
func (f *Forest) ApplyFeatureSource(list FeatureSource) ActionParams {
	params := append(ActionParams{}, f.Base...)
	for i, tree := range f.Trees {
		w := f.Weights[i]
		for j, x := range tree.FindFeatureSource(list) {
			params[j] += x * w
		}
	}
	return params
}

func (f *Forest) applyBatch(in anyvec.Vector, batch int) anyvec.Vector {
	features := vecToFloats(in)
	numFeatures := len(features) / batch

	var outParams []float64
	for i := 0; i < batch; i++ {
		subFeatures := features[i*numFeatures : (i+1)*numFeatures]
		params := f.Apply(subFeatures)
		outParams = append(outParams, params...)
	}

	c := in.Creator()
	vecData := c.MakeNumericList(outParams)
	return c.MakeVectorData(vecData)
}

func (f *Forest) applySamples(samples []Sample) []ActionParams {
	res := make([]ActionParams, len(samples))
	indices := make(chan int, len(samples))
	for i := range samples {
		indices <- i
	}
	close(indices)
	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range indices {
				res[i] = f.ApplyFeatureSource(samples[i])
			}
		}()
	}
	wg.Wait()
	return res
}

// Tree is a node in a decision tree.
//
// A Tree is either a leaf node or a branching node.
type Tree struct {
	Leaf bool `json:",omitempty"`

	// Information for leaf nodes.
	Params ActionParams `json:",omitempty"`

	// Information for branching nodes.
	Feature      int     `json:",omitempty"`
	Threshold    float64 `json:",omitempty"`
	LessThan     *Tree   `json:",omitempty"`
	GreaterEqual *Tree   `json:",omitempty"`
}

// Find finds the leaf parameters for the features.
func (t *Tree) Find(features []float64) ActionParams {
	return t.FindFeatureSource(sliceFeatureSource(features))
}

// FindFeatureSource is like Find, but for a
// FeatureSource.
func (t *Tree) FindFeatureSource(list FeatureSource) ActionParams {
	if t.Leaf {
		return t.Params
	}
	val := list.Feature(t.Feature)
	if val < t.Threshold {
		return t.LessThan.FindFeatureSource(list)
	} else {
		return t.GreaterEqual.FindFeatureSource(list)
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

type sliceFeatureSource []float64

func (s sliceFeatureSource) NumFeatures() int {
	return len(s)
}

func (s sliceFeatureSource) Feature(i int) float64 {
	return s[i]
}
