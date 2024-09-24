package processing

import (
	"fmt"
	"github.com/okieraised/go-faceid-pipeline/utils"
	"gorgonia.org/tensor"
)

func NMS(dets *tensor.Dense, threshold float32) ([]int, error) {
	x1, err := dets.Slice(nil, tensor.S(0))
	if err != nil {
		return nil, err
	}

	x1Owned := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(x1.(*tensor.Dense).Shape()...),
	)

	err = tensor.Copy(x1Owned, x1)
	if err != nil {
		return nil, err
	}

	y1, err := dets.Slice(nil, tensor.S(1))
	if err != nil {
		return nil, err
	}

	y1Owned := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(y1.(*tensor.Dense).Shape()...),
	)

	err = tensor.Copy(y1Owned, y1)
	if err != nil {
		return nil, err
	}

	x2, err := dets.Slice(nil, tensor.S(2))
	if err != nil {
		return nil, err
	}

	x2Owned := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(x2.(*tensor.Dense).Shape()...),
	)

	err = tensor.Copy(x2Owned, x2)
	if err != nil {
		return nil, err
	}

	y2, err := dets.Slice(nil, tensor.S(3))
	if err != nil {
		return nil, err
	}

	y2Owned := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(y2.(*tensor.Dense).Shape()...),
	)

	err = tensor.Copy(y2Owned, y2)
	if err != nil {
		return nil, err
	}

	scores, err := dets.Slice(nil, tensor.S(4))
	if err != nil {
		return nil, err
	}

	scoresOwned := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(scores.Shape()...),
	)

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

	// Sort scores in descending order
	order, err := utils.ArgSortDescending(scoresOwned)
	if err != nil {
		return nil, err
	}

	fmt.Println("areas", areas)
	fmt.Println("order", order)

	keep := []int{}
	for len(order) > 0 {
		i := order[0]
		keep = append(keep, i)

		x1i, err := x1Owned.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}

		fmt.Println("x1Owned", x1Owned.Shape())

		x11, err := utils.SelectRows3D(x1Owned, order[1:])
		if err != nil {
			return nil, err
		}

		fmt.Println("x1i", x1i)
		fmt.Println("x11", x11)

		break

		//xx1, _ := tensor.Maximum(x1.Slice(tensor.S(i)), x1.Slice(tensor.S(orderSlice...)))
		//yy1, _ := tensor.Maximum(y1.Slice(tensor.S(i)), y1.Slice(tensor.S(orderSlice...)))
		//xx2, _ := tensor.Minimum(x2.Slice(tensor.S(i)), x2.Slice(tensor.S(orderSlice...)))
		//yy2, _ := tensor.Minimum(y2.Slice(tensor.S(i)), y2.Slice(tensor.S(orderSlice...)))
		//
		//w, _ := tensor.Sub(xx2, xx1)
		//wPlus1, _ := tensor.Add(w, constantOne)
		//h, _ := tensor.Sub(yy2, yy1)
		//hPlus1, _ := tensor.Add(h, constantOne)
		//
		//// Clamp to 0 (no negative width/height)
		//wClamped, _ := tensor.Maximum(wPlus1, constantOne)
		//hClamped, _ := tensor.Maximum(hPlus1, constantOne)
		//
		//// Calculate intersection area
		//inter, _ := tensor.Mul(wClamped, hClamped)
		//
		//// Compute overlap (IoU)
		//areasI, _ := areas.At(i)
		//areasRemaining, _ := areas.Slice(tensor.S(orderSlice...))
		//union, _ := tensor.Sub(areasRemaining, inter)
		//iou, _ := tensor.Div(inter, union)
		//
		//// Keep indices where overlap <= thresh
		//ovrData := iou.Data().([]float32)
		//inds := []int{}
		//for idx, ovr := range ovrData {
		//	if ovr <= threshold {
		//		inds = append(inds, idx)
		//	}
		//}
		//
		//newOrder := make([]int, len(inds))
		//for k, v := range inds {
		//	newOrder[k] = order[v+1]
		//}
		//order = newOrder
	}

	return keep, nil
}
