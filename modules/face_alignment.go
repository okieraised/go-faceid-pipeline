package modules

import (
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/utils"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
	"math"
)

type FaceAlignmentClient struct {
	imageSize         [2]int
	standardLandmarks *tensor.Dense
}

func NewFaceAlignmentClient(cfg *config.FaceAlignParams) *FaceAlignmentClient {
	return &FaceAlignmentClient{
		imageSize:         cfg.ImageSize,
		standardLandmarks: cfg.StandardLandmarks,
	}
}

func (c *FaceAlignmentClient) Infer(img gocv.Mat, bbox, landmarks *tensor.Dense) (*gocv.Mat, error) {

	from, err := utils.TensorToPoint2fVector(landmarks)
	if err != nil {
		return nil, err
	}

	to, err := utils.TensorToPoint2fVector(c.standardLandmarks)
	if err != nil {
		return nil, err
	}

	inliers := gocv.NewMat()
	affineMatrix := gocv.EstimateAffinePartial2DWithParams(
		from,
		to,
		inliers,
		int(gocv.HomograpyMethodLMEDS),
		3.0,
		2000,
		0.99,
		10,
	)
	err = inliers.Close()
	if err != nil {
		return nil, err
	}

	alignedImg := gocv.NewMat()

	if affineMatrix.Empty() {
		var det *tensor.Dense
		if bbox == nil {
			det = tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(4),
				tensor.WithBacking([]float32{0, 0, 0, 0}),
			)
			det0 := float32(img.Size()[1]) * 0.0625
			det1 := float32(img.Size()[0]) * 0.0625
			det2 := float32(img.Size()[1]) - det0
			det3 := float32(img.Size()[0]) - det1

			det.Set(0, det0)
			det.Set(1, det1)
			det.Set(2, det2)
			det.Set(3, det3)
		} else {
			det = bbox
		}
		var margin float32 = 44
		bb := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(4),
			tensor.WithBacking([]float32{0, 0, 0, 0}),
		)

		bb.Set(0, float32(math.Max(float64(det.GetF32(0)-margin/2), 0)))
		bb.Set(1, float32(math.Max(float64(det.GetF32(1)-margin/2), 0)))
		bb.Set(2, float32(math.Max(float64(det.GetF32(2)+margin/2), float64(img.Size()[1]))))
		bb.Set(3, float32(math.Max(float64(det.GetF32(3)+margin/2), float64(img.Size()[0]))))

		x0 := bb.GetF32(0)
		y0 := bb.GetF32(1)
		x1 := bb.GetF32(2)
		y1 := bb.GetF32(3)
		width := int(x1) - int(x0)
		height := int(y1) - int(y0)

		roi := img.Region(image.Rect(int(x0), int(y0), width, height))
		gocv.Resize(roi, &alignedImg, image.Point{X: c.imageSize[0], Y: c.imageSize[1]}, 0, 0, gocv.InterpolationLinear)

	} else {
		gocv.WarpAffine(
			img,
			&alignedImg,
			affineMatrix,
			image.Point{
				X: c.imageSize[0],
				Y: c.imageSize[1],
			},
		)

	}
	return &alignedImg, nil
}
