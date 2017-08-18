package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"math"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/treeagent"
)

func main() {
	var forestFile string
	var envName string
	var heatmapOut string
	flag.StringVar(&forestFile, "in", "", "forest json file")
	flag.StringVar(&envName, "env", "", "muniverse environment name")
	flag.StringVar(&heatmapOut, "heatmap", "heatmap.png", "heatmap output file")
	flag.Parse()

	if forestFile == "" {
		essentials.Die("Missing -in flag. See -help.")
	}

	data, err := ioutil.ReadFile(forestFile)
	if err != nil {
		essentials.Die(err)
	}
	var forest *treeagent.Forest
	if err := json.Unmarshal(data, &forest); err != nil {
		essentials.Die(err)
	}

	fmt.Println("   # trees:", len(forest.Trees))

	counts := countFeatures(forest)
	fmt.Println("# features:", len(counts))

	if envName == "" {
		fmt.Println("No -env flag; skipping pictures.")
		return
	}

	spec := muniverse.SpecForName(envName)
	if spec == nil {
		essentials.Die("Environment not found:", envName)
	}
	heatmap := featureHeatmap(counts, spec)
	f, err := os.Create(heatmapOut)
	if err != nil {
		essentials.Die(err)
	}
	defer f.Close()
	if err := png.Encode(f, heatmap); err != nil {
		essentials.Die(err)
	}
}

func countFeatures(f *treeagent.Forest) map[int]int {
	counts := map[int]int{}
	var addTree func(t *treeagent.Tree)
	addTree = func(t *treeagent.Tree) {
		if !t.Leaf {
			counts[t.Feature]++
			addTree(t.LessThan)
			addTree(t.GreaterEqual)
		}
	}
	for _, tree := range f.Trees {
		addTree(tree)
	}
	return counts
}

func featureHeatmap(counts map[int]int, e *muniverse.EnvSpec) image.Image {
	width, height := muniverseDims(e)
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	max := maxCount(counts)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			featureIdx := x + y*width
			heat := math.Log(float64(counts[featureIdx])) / math.Log(float64(max))
			heatByte := uint8(heat * 0xff)
			fmt.Println(heatByte)
			img.SetRGBA(x, y, color.RGBA{
				R: heatByte,
				A: 0xff,
			})
		}
	}

	return img
}

func maxCount(featureCounts map[int]int) int {
	var max int
	for _, count := range featureCounts {
		max = essentials.MaxInt(count, max)
	}
	return max
}

func muniverseDims(e *muniverse.EnvSpec) (width, height int) {
	width = e.Width / 4
	height = e.Height / 4
	if e.Width%4 != 0 {
		width++
	}
	if e.Height%4 != 0 {
		height++
	}
	return
}
