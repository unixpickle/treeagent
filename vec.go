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

func numToFloat(num anyvec.Numeric) float64 {
	switch num := num.(type) {
	case float64:
		return num
	case float32:
		return float64(num)
	default:
		panic("unsupported numeric type")
	}
}

// smallVec is a native vector type optimized to be used
// with a small number of components.
//
// Most smallVec methods return the receiver so that
// vector operations can be chained more easily.
type smallVec []float64

func (s smallVec) Copy() smallVec {
	return append(smallVec{}, s...)
}

func (s smallVec) Scale(scale float64) smallVec {
	for i, x := range s {
		s[i] = x * scale
	}
	return s
}

func (s smallVec) Add(other smallVec) smallVec {
	for i, x := range other {
		s[i] += x
	}
	return s
}

func (s smallVec) Sub(other smallVec) smallVec {
	for i, x := range other {
		s[i] -= x
	}
	return s
}

func (s smallVec) Dot(other smallVec) float64 {
	var res float64
	for i, x := range s {
		res += x * other[i]
	}
	return res
}

func (s smallVec) AbsSum() float64 {
	var res float64
	for _, x := range s {
		if x < 0 {
			res -= x
		} else {
			res += x
		}
	}
	return res
}
