package processing

import (
	"gorgonia.org/tensor"
	"math"
)

func ClipBoxes(boxes *tensor.Dense, imgShape []int) (*tensor.Dense, error) {
	width := float32(imgShape[1] - 1)
	height := float32(imgShape[0] - 1)

	boxes04, err := boxes.Slice(nil, tensor.S(0, boxes.Shape()[1]-1, 4))
	if err != nil {
		return nil, err
	}
	res04, err := boxes04.(*tensor.Dense).Apply(func(x float32) float32 {
		return float32(math.Max(math.Min(float64(x), float64(width)), 0))
	})

	err = tensor.Copy(boxes04, res04)
	if err != nil {
		return nil, err
	}

	boxes14, err := boxes.Slice(nil, tensor.S(1, boxes.Shape()[1]-1, 4))
	if err != nil {
		return nil, err
	}
	res14, err := boxes14.(*tensor.Dense).Apply(func(x float32) float32 {
		return float32(math.Max(math.Min(float64(x), float64(height)), 0))
	})
	err = tensor.Copy(boxes14, res14)
	if err != nil {
		return nil, err
	}

	boxes24, err := boxes.Slice(nil, tensor.S(2, boxes.Shape()[1]-1, 4))
	if err != nil {
		return nil, err
	}
	res24, err := boxes24.(*tensor.Dense).Apply(func(x float32) float32 {
		return float32(math.Max(math.Min(float64(x), float64(width)), 0))
	})
	err = tensor.Copy(boxes24, res24)
	if err != nil {
		return nil, err
	}

	boxes34, err := boxes.Slice(nil, tensor.S(3, boxes.Shape()[1]-1, 4))
	if err != nil {
		return nil, err
	}
	res34, err := boxes34.(*tensor.Dense).Apply(func(x float32) float32 {
		return float32(math.Max(math.Min(float64(x), float64(height)), 0))
	})
	err = tensor.Copy(boxes34, res34)
	if err != nil {
		return nil, err
	}

	return boxes, nil
}
