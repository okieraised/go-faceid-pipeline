package processing

import (
	"github.com/okieraised/go-faceid-pipeline/utils"
	"gorgonia.org/tensor"
)

func NMS(dets *tensor.Dense, threshold float32) ([]int, error) {
	x1, err := dets.Slice(nil, tensor.S(0))
	if err != nil {
		return nil, err
	}
	x1Owned := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(x1.(*tensor.Dense).Shape()...))

	err = tensor.Copy(x1Owned, x1)
	if err != nil {
		return nil, err
	}

	y1, err := dets.Slice(nil, tensor.S(1))
	if err != nil {
		return nil, err
	}
	y1Owned := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(y1.(*tensor.Dense).Shape()...))
	err = tensor.Copy(y1Owned, y1)
	if err != nil {
		return nil, err
	}

	x2, err := dets.Slice(nil, tensor.S(2))
	if err != nil {
		return nil, err
	}

	x2Owned := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(x2.(*tensor.Dense).Shape()...))

	err = tensor.Copy(x2Owned, x2)
	if err != nil {
		return nil, err
	}

	y2, err := dets.Slice(nil, tensor.S(3))
	if err != nil {
		return nil, err
	}

	y2Owned := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(y2.(*tensor.Dense).Shape()...))

	err = tensor.Copy(y2Owned, y2)
	if err != nil {
		return nil, err
	}

	scores, err := dets.Slice(nil, tensor.S(4))
	if err != nil {
		return nil, err
	}

	scoresOwned := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(scores.Shape()...))

	err = tensor.Copy(scoresOwned, scores)
	if err != nil {
		return nil, err
	}

	x2SubX1, err := x2Owned.Sub(x1Owned)
	if err != nil {
		return nil, err
	}
	y2SubY1, err := y2Owned.Sub(y1Owned)
	if err != nil {
		return nil, err
	}

	wPlus1, err := x2SubX1.AddScalar(float32(1), true)
	if err != nil {
		return nil, err
	}
	hPlus1, err := y2SubY1.AddScalar(float32(1), true)
	if err != nil {
		return nil, err
	}

	areas, err := wPlus1.Mul(hPlus1)
	if err != nil {
		return nil, err
	}

	order, err := utils.ArgSortDescending(scoresOwned)
	if err != nil {
		return nil, err
	}
	keep := make([]int, 0)

	for len(order) > 0 {
		i := order[0]
		keep = append(keep, i)

		x1i, err := x1Owned.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}

		x11, err := utils.TensorByIndices(x1Owned, order[1:])
		if err != nil {
			return nil, err
		}
		xx1, err := tensor.MaxBetween(x1i, x11)
		if err != nil {
			return nil, err
		}

		y1i, err := y1Owned.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}

		y11, err := utils.TensorByIndices(y1Owned, order[1:])
		//y11, err := y1Owned.Slice(tensor.S(order[1], len(order)))
		if err != nil {
			return nil, err
		}
		yy1, err := tensor.MaxBetween(y1i, y11)
		if err != nil {
			return nil, err
		}

		x2i, err := x2Owned.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}

		x22, err := utils.TensorByIndices(x2Owned, order[1:])
		if err != nil {
			return nil, err
		}
		xx2, err := tensor.MinBetween(x2i, x22)
		if err != nil {
			return nil, err
		}

		y2i, err := y2Owned.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}

		y22, err := utils.TensorByIndices(y2Owned, order[1:])
		if err != nil {
			return nil, err
		}
		yy2, err := tensor.MinBetween(y2i, y22)
		if err != nil {
			return nil, err
		}

		w, err := xx2.(*tensor.Dense).Sub(xx1.(*tensor.Dense))
		if err != nil {
			return nil, err
		}

		wPlus2, err := w.AddScalar(float32(1), true)
		if err != nil {
			return nil, err
		}

		h, err := yy2.(*tensor.Dense).Sub(yy1.(*tensor.Dense))
		if err != nil {
			return nil, err
		}
		hPlus2, err := h.AddScalar(float32(1), true)
		if err != nil {
			return nil, err
		}

		wClamped, err := tensor.MaxBetween(float32(0), wPlus2)
		if err != nil {
			return nil, err
		}

		hClamped, err := tensor.MaxBetween(float32(0), hPlus2)
		if err != nil {
			return nil, err
		}

		inter, err := wClamped.(*tensor.Dense).Mul(hClamped.(*tensor.Dense))
		if err != nil {
			return nil, err
		}

		areasI, err := areas.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}

		areasRemaining, err := utils.TensorByIndices(areas, order[1:])
		if err != nil {
			return nil, err
		}

		areaSum, err := tensor.Add(areasI, areasRemaining)
		if err != nil {
			return nil, err
		}

		union, err := areaSum.(*tensor.Dense).Sub(inter)
		if err != nil {
			return nil, err
		}

		iou, err := inter.Div(union)
		if err != nil {
			return nil, err
		}
		ovrData := iou.Float32s()

		indices := make([]int, 0)
		for idx, ovr := range ovrData {
			if ovr <= threshold {
				indices = append(indices, idx)
			}
		}
		newOrder := make([]int, 0)
		for _, v := range indices {
			newOrder = append(newOrder, order[v+1])
		}
		order = newOrder
	}

	return keep, nil
}
