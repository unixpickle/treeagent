package treeagent

import "github.com/unixpickle/anyvec"

func vecToFloats(vec anyvec.Vector) []float64 {
	var res []float64
	switch data := vec.Data().(type) {
	case []float64:
		res = data
	case []float32:
		for _, x := range data {
			res = append(res, float64(x))
		}
	default:
		panic("unsupported numeric type")
	}
	return res
}
