package rcnn

import "gorgonia.org/tensor"

func Anchors(height, width, stride int, baseAnchor *tensor.Dense) (*tensor.Dense, error) {

	a := baseAnchor.Shape()[0]

	allAnchors := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(height, width, a, 4),
	)

	for iw := range width {
		sw := float32(iw * stride)
		for ih := range height {
			sh := float32(ih * stride)
			for k := range a {
				val0, err := baseAnchor.At(k, 0)
				if err != nil {
					return nil, err
				}
				val1, err := baseAnchor.At(k, 1)
				if err != nil {
					return nil, err
				}
				val2, err := baseAnchor.At(k, 2)
				if err != nil {
					return nil, err
				}
				val3, err := baseAnchor.At(k, 3)
				if err != nil {
					return nil, err
				}

				err = allAnchors.SetAt(val0.(float32)+sw, ih, iw, k, 0)
				if err != nil {
					return nil, err
				}
				err = allAnchors.SetAt(val1.(float32)+sh, ih, iw, k, 1)
				if err != nil {
					return nil, err
				}
				err = allAnchors.SetAt(val2.(float32)+sw, ih, iw, k, 2)
				if err != nil {
					return nil, err
				}
				err = allAnchors.SetAt(val3.(float32)+sh, ih, iw, k, 3)
				if err != nil {
					return nil, err
				}
			}
		}
	}

	return allAnchors, nil
}
