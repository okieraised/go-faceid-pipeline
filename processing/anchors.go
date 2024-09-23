package processing

import (
	"errors"
	"fmt"
	"github.com/okieraised/go-faceid-pipeline/utils"
	"gorgonia.org/tensor"
	"sort"
	"strconv"
)

type AnchorConfig struct {
	BaseSize      int
	Ratios        []float32
	Scales        []float32
	AllowedBorder int
}

func GenerateAnchorsFPN2(denseAnchor bool, cfg map[string]AnchorConfig) ([]*tensor.Dense, error) {

	rpnFeatStride := make([]int, 0)
	for k := range cfg {
		kAsInt, err := strconv.Atoi(k)
		if err != nil {
			return nil, err
		}
		rpnFeatStride = append(rpnFeatStride, kAsInt)
	}
	sort.Slice(rpnFeatStride, func(i, j int) bool {
		return rpnFeatStride[i] > rpnFeatStride[j]
	})

	anchors := make([]*tensor.Dense, 0)
	for _, k := range rpnFeatStride {
		v := cfg[strconv.Itoa(k)]
		bs := v.BaseSize
		ratios := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(1, len(v.Ratios)),
			tensor.WithBacking(v.Ratios),
		)
		scales := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(1, len(v.Scales)),
			tensor.WithBacking(v.Scales),
		)
		r, err := generateAnchors2(bs, ratios, scales, k, utils.RefPointer(denseAnchor))
		if err != nil {
			return nil, err
		}
		anchors = append(anchors, r)
	}
	return anchors, nil
}

func generateAnchors2(baseSize int, ratios, scales *tensor.Dense, stride int, denseAnchor *bool) (*tensor.Dense, error) {
	if denseAnchor == nil {
		denseAnchor = utils.RefPointer(false)
	}

	baseAnchors := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(1, 4),
		tensor.WithBacking([]float32{1 - 1, 1 - 1, float32(baseSize) - 1, float32(baseSize) - 1}),
	)

	ratioAnchors, err := ratioEnum(baseAnchors, ratios)
	if err != nil {
		return nil, err
	}

	scaledAnchors := make([]*tensor.Dense, 0)
	for i := range ratioAnchors.Shape()[0] {
		sliced, err := ratioAnchors.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}
		scaled, err := scaleEnum(sliced.(*tensor.Dense), scales)
		if err != nil {
			return nil, err
		}
		scaledAnchors = append(scaledAnchors, scaled)
	}
	anchors, err := scaledAnchors[0].Vstack(scaledAnchors[1:]...)
	if utils.DerefPointer(denseAnchor) {
		if stride%2 != 0 {
			return nil, errors.New(fmt.Sprintf("stride must be even number, got %d", stride))
		}

		anchors2 := anchors.Clone().(*tensor.Dense)
		err = tensor.Copy(anchors2, anchors)
		if err != nil {
			return nil, err
		}
		ret, err := anchors2.AddScalar(float32(stride/2), true)
		if err != nil {
			return nil, err
		}
		anchors, err = anchors.Vstack(ret)
		if err != nil {
			return nil, err
		}
	}

	return anchors, nil
}

func ratioEnum(anchor, ratios *tensor.Dense) (*tensor.Dense, error) {

	w, h, centerX, centerY := whctrs(anchor)
	size := w * h
	sizeRatios, err := ratios.DivScalar(size, false)
	if err != nil {
		return nil, err
	}

	ws, err := tensor.Sqrt(sizeRatios)
	if err != nil {
		return nil, err
	}

	hs, err := ws.(*tensor.Dense).Mul(ratios)
	if err != nil {
		return nil, err
	}
	anchors, err := mkanchors(ws.(*tensor.Dense), hs, centerX, centerY)
	if err != nil {
		return nil, err
	}

	return anchors, nil
}

func whctrs(anchor *tensor.Dense) (float32, float32, float32, float32) {

	rawAnchors := anchor.Float32s()

	w := rawAnchors[2] - rawAnchors[0] + 1
	h := rawAnchors[3] - rawAnchors[1] + 1
	centerX := rawAnchors[0] + 0.5*(w-1)
	centerY := rawAnchors[1] + 0.5*(h-1)

	return w, h, centerX, centerY

}

func mkanchors(ws, hs *tensor.Dense, centerX, centerY float32) (*tensor.Dense, error) {

	wsShape := ws.Shape()
	err := ws.Reshape(wsShape[1], 1)
	if err != nil {
		return nil, err
	}

	hsShape := hs.Shape()
	err = hs.Reshape(hsShape[1], 1)
	if err != nil {
		return nil, err
	}
	ws, err = ws.SubScalar(float32(1.0), true)
	if err != nil {
		return nil, err
	}
	ws, err = ws.MulScalar(float32(0.5), true)
	if err != nil {
		return nil, err
	}
	anchor0, err := ws.SubScalar(centerX, false)
	if err != nil {
		return nil, err
	}
	anchor2, err := ws.AddScalar(centerX, false)
	if err != nil {
		return nil, err
	}
	hs, err = hs.SubScalar(float32(1.0), true)
	if err != nil {
		return nil, err
	}
	hs, err = hs.MulScalar(float32(0.5), true)
	if err != nil {
		return nil, err
	}
	anchor1, err := hs.SubScalar(centerY, false)
	if err != nil {
		return nil, err
	}
	anchor3, err := hs.AddScalar(centerY, false)
	if err != nil {
		return nil, err
	}

	anchors, err := anchor0.Hstack(anchor1, anchor2, anchor3)
	if err != nil {
		return nil, err
	}

	return anchors, nil
}

func scaleEnum(anchor, scales *tensor.Dense) (*tensor.Dense, error) {

	w, h, centerX, centerY := whctrs(anchor)

	ws, err := scales.MulScalar(w, true)
	if err != nil {
		return nil, err
	}

	hs, err := scales.MulScalar(h, true)
	if err != nil {
		return nil, err
	}

	anchors, err := mkanchors(ws, hs, centerX, centerY)
	if err != nil {
		return nil, err
	}

	return anchors, nil
}
