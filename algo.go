package treeagent

import "math"

// A TreeAlgorithm is an algorithm for building trees.
//
// Different tree algorithms solve different objectives.
// Thus, different step sizes may be best for different
// tree algorithms.
type TreeAlgorithm int

// TreeAlgorithms contains all supported TreeAlgorithms.
var TreeAlgorithms = []TreeAlgorithm{
	SumAlgorithm,
	MeanAlgorithm,
	MSEAlgorithm,
	BalancedSumAlgorithm,
	StddevAlgorithm,
	SignAlgorithm,
	AbsAlgorithm,
}

const (
	// SumAlgorithm constructs a tree where the leaf nodes
	// contain gradient sums for the repersented samples.
	SumAlgorithm TreeAlgorithm = iota

	// MeanAlgorithm is similar to SumAlgorithm, except
	// that gradients are averaged instead of summed.
	MeanAlgorithm

	// MSEAlgorithm constructs a tree by minimizing
	// mean-squared error over gradients.
	MSEAlgorithm

	// BalancedSumAlgorithm is similar to SumAlgorithm,
	// but splits are biased towards balanced trees.
	BalancedSumAlgorithm

	// StddevAlgorithm has the same objective as MSE and
	// mean, but it uses a splitting criteria based on
	// gradient standard deviations.
	StddevAlgorithm

	// SignAlgorithm maximizes the dot products between
	// sums of gradients and the sums' sign.
	// The resulting leaf parameters have values 0, 1, or
	// -1.
	SignAlgorithm

	// AbsAlgorithm uses the same splitting criteria as
	// SignAlgorithm, but it uses the gradient means in
	// the leaves instead of the gradient signs.
	AbsAlgorithm
)

// String returns a human-readable representation of the
// algorithm, like "mse" or "abs".
func (t TreeAlgorithm) String() string {
	switch t {
	case SumAlgorithm:
		return "sum"
	case MeanAlgorithm:
		return "mean"
	case MSEAlgorithm:
		return "mse"
	case BalancedSumAlgorithm:
		return "balancedsum"
	case StddevAlgorithm:
		return "stddev"
	case SignAlgorithm:
		return "sign"
	case AbsAlgorithm:
		return "abs"
	default:
		return ""
	}
}

func (t TreeAlgorithm) splitTracker() splitTracker {
	switch t {
	case SumAlgorithm:
		return &sumTracker{}
	case MeanAlgorithm:
		return &meanTracker{}
	case MSEAlgorithm:
		return &mseTracker{}
	case BalancedSumAlgorithm:
		return &balancedSumTracker{}
	case StddevAlgorithm:
		return &stddevTracker{}
	case SignAlgorithm, AbsAlgorithm:
		return &signTracker{}
	default:
		panic("unknown tree algorithm")
	}
}

func (t TreeAlgorithm) leafParams(leafData, allData []*gradientSample) smallVec {
	switch t {
	case SignAlgorithm:
		return sumGradients(leafData).Signs()
	case SumAlgorithm, BalancedSumAlgorithm:
		return sumGradients(leafData).Scale(1 / float64(len(allData)))
	case MeanAlgorithm, MSEAlgorithm, StddevAlgorithm, AbsAlgorithm:
		return sumGradients(leafData).Scale(1 / float64(len(leafData)))
	default:
		panic("unknown tree algorithm")
	}
}

// A splitTracker dynamically computes how good splits are
// on a spectrum of possible splits.
type splitTracker interface {
	Reset(rightSamples []*gradientSample)
	MoveToLeft(sample *gradientSample)
	Quality() float64
}

// A sumTracker is a splitTracker for SumAlgorithm.
type sumTracker struct {
	leftSum  smallVec
	rightSum smallVec
}

func (s *sumTracker) Reset(rightSamples []*gradientSample) {
	s.rightSum = sumGradients(rightSamples)
	s.leftSum = make(smallVec, len(s.rightSum))
}

func (s *sumTracker) MoveToLeft(sample *gradientSample) {
	s.rightSum.Sub(sample.Gradient)
	s.leftSum.Add(sample.Gradient)
}

func (s *sumTracker) Quality() float64 {
	var sum float64
	for _, vec := range []smallVec{s.leftSum, s.rightSum} {
		sum += vec.Dot(vec)
	}
	return sum
}

// A balancedSumTracker is a splitTracker for
// BalancedSumAlgorithm.
type balancedSumTracker struct {
	sumTracker sumTracker
	leftCount  int
	rightCount int
}

func (b *balancedSumTracker) Reset(rightSamples []*gradientSample) {
	b.sumTracker.Reset(rightSamples)
	b.leftCount = 0
	b.rightCount = len(rightSamples)
}

func (b *balancedSumTracker) MoveToLeft(sample *gradientSample) {
	b.sumTracker.MoveToLeft(sample)
	b.leftCount++
	b.rightCount--
}

func (b *balancedSumTracker) Quality() float64 {
	return b.sumTracker.Quality() * float64(b.leftCount*b.rightCount)
}

// A meanTracker is a splitTracker for MeanAlgorithm.
type meanTracker struct {
	sumTracker sumTracker
	leftCount  int
	rightCount int
}

func (m *meanTracker) Reset(rightSamples []*gradientSample) {
	m.sumTracker.Reset(rightSamples)
	m.leftCount = 0
	m.rightCount = len(rightSamples)
}

func (m *meanTracker) MoveToLeft(sample *gradientSample) {
	m.sumTracker.MoveToLeft(sample)
	m.leftCount++
	m.rightCount--
}

func (m *meanTracker) Quality() float64 {
	s := &m.sumTracker
	sums := []smallVec{s.leftSum, s.rightSum}
	counts := []int{m.leftCount, m.rightCount}

	var sum float64
	for i, vec := range sums {
		sum += vec.Dot(vec) / float64(counts[i])
	}

	return sum
}

// A mseTracker is a splitTracker for MSEAlgorithm.
type mseTracker struct {
	sumTracker   sumTracker
	leftSquares  float64
	rightSquares float64
	leftCount    int
	rightCount   int
}

func (m *mseTracker) Reset(rightSamples []*gradientSample) {
	m.sumTracker.Reset(rightSamples)
	m.leftSquares = 0
	m.rightSquares = 0
	for _, sample := range rightSamples {
		m.rightSquares += sample.Gradient.Dot(sample.Gradient)
	}
	m.leftCount = 0
	m.rightCount = len(rightSamples)
}

func (m *mseTracker) MoveToLeft(sample *gradientSample) {
	m.sumTracker.MoveToLeft(sample)
	sq := sample.Gradient.Dot(sample.Gradient)
	m.leftSquares += sq
	m.rightSquares -= sq
	m.leftCount++
	m.rightCount--
}

func (m *mseTracker) Quality() float64 {
	left, right := m.leftRightErrors()
	return -(left + right)
}

func (m *mseTracker) leftRightErrors() (left, right float64) {
	// The minimal MSE is equivalent to
	//
	//     Var(x) = E[X^2] - E^2[X]
	//
	// Scaling this by n, we get:
	//
	//     Error = (x1^2 + ... + xn^2) - (x1 + ... + xn)^2/n
	//

	sums := []smallVec{m.sumTracker.leftSum, m.sumTracker.rightSum}
	sqSums := []float64{m.leftSquares, m.rightSquares}
	counts := []int{m.leftCount, m.rightCount}

	reses := make([]float64, 2)
	for i, sum := range sums {
		n := float64(counts[i])
		if n == 0 {
			continue
		}
		reses[i] = sqSums[i] - sum.Dot(sum)/n
	}

	return reses[0], reses[1]
}

// stddevTracker is a splitTracker for StddevAlgorithm.
type stddevTracker struct {
	mseTracker
}

func (s *stddevTracker) Quality() float64 {
	// Equivalent to minimizing N1*stddev1 + N2*stddev2
	left, right := s.leftRightErrors()
	return -(math.Sqrt(float64(s.leftCount)*left) +
		math.Sqrt(float64(s.rightCount)*right))
}

// signTracker is a splitTracker for SignAlgorithm.
type signTracker struct {
	sumTracker
}

func (s *signTracker) Quality() float64 {
	return s.leftSum.AbsSum() + s.rightSum.AbsSum()
}
