package modules

import (
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/utils"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"math"
)

type FaceSelectionClient struct {
	*config.FaceSelectionParams
}

func NewFaceSelectionClient(cfg *config.FaceSelectionParams) *FaceSelectionClient {
	return &FaceSelectionClient{
		FaceSelectionParams: cfg,
	}
}

func (c *FaceSelectionClient) Infer(img gocv.Mat, faceBoxes, faceKeyPoints *tensor.Dense, isEnroll *bool) (*tensor.Dense, *tensor.Dense, error) {
	if isEnroll == nil {
		isEnroll = utils.RefPointer(false)
	}

	// If enroll
	if utils.DerefPointer(isEnroll) {
		biggestBox, biggestKeyPoint, err := c.getBiggestAreaFace(faceBoxes, faceKeyPoints)
		if err != nil {
			return nil, nil, err
		}
		if biggestBox != nil && c.isFaceAreaBigEnough(img, biggestBox) {
			return biggestBox, biggestKeyPoint, nil
		} else {
			return nil, nil, nil
		}
	}

	imgShape := img.Size()

	marginCenterLeft := c.MarginCenterLeftRatio * float32(imgShape[1])
	marginCenterRight := c.MarginCenterRightRatio * float32(imgShape[1])
	marginEdge := float32(math.Min(50, float64(c.MarginEdgeRatio*float32(imgShape[1]))))

	xCen := imgShape[1] / 2

	detections := faceBoxes.Clone().(*tensor.Dense)
	var outBBoxes *tensor.Dense
	validBoxes := make([]*tensor.Dense, 0)
	detSize := detections.Shape()[0]

	for i := 0; i < detSize; i++ {
		ret, err := detections.Slice(tensor.S(i))
		if err != nil {
			return nil, nil, err
		}
		result := ret.(*tensor.Dense)

		area := (result.GetF32(2) - result.GetF32(0)) * (result.GetF32(2) - result.GetF32(0))
		boxCenterWidth := (result.GetF32(0) + result.GetF32(2)) / 2
		boxCenterHeight := (result.GetF32(1) + result.GetF32(3)) / 2

		if (boxCenterWidth >= marginEdge) &&
			(boxCenterWidth <= float32(imgShape[1])-marginEdge) &&
			(boxCenterHeight >= marginEdge) &&
			(boxCenterHeight <= float32(imgShape[0])-marginEdge) &&
			(area/(float32(imgShape[0])*float32(imgShape[1])) >= c.MinimumFaceRatio) {
			validBoxes = append(validBoxes, result)
		}
	}

	centerBoxes := make([]*tensor.Dense, 0)
	//centerLandmarks := make([]*tensor.Dense, 0)

	for _, result := range validBoxes {
		boxesCenterWidth := (result.GetF32(0) + result.GetF32(2)) / 2
		if -marginCenterLeft <= boxesCenterWidth-float32(xCen) && boxesCenterWidth-float32(xCen) <= marginCenterRight {
			centerBoxes = append(centerBoxes, result)
		}
	}

	if len(centerBoxes) == 0 {
		if len(validBoxes) == 0 {
			centerBoxes = []*tensor.Dense{detections}
		} else {
			centerBoxes = validBoxes
		}
	}

	var maxSize float32 = 0
	for _, result := range centerBoxes {
		temSize := (result.GetF32(2) - result.GetF32(0)) + (result.GetF32(3) - result.GetF32(1))
		if temSize > maxSize {
			outBBoxes = result.Clone().(*tensor.Dense)
		}
	}
	if outBBoxes == nil {
		return nil, nil, nil
	}

	var outKeypoint *tensor.Dense

	for i := 0; i < faceBoxes.Shape()[0]; i++ {
		bboxS, err := faceBoxes.Slice(tensor.S(i))
		if err != nil {
			return nil, nil, err
		}
		bbox := bboxS.(*tensor.Dense)

		keyPointS, err := faceKeyPoints.Slice(tensor.S(i))
		if err != nil {
			return nil, nil, err
		}
		keyPoint := keyPointS.(*tensor.Dense)

		x, y, x2, y2 := bbox.GetF32(0), bbox.GetF32(1), bbox.GetF32(2), bbox.GetF32(3)
		xMinOut, yMinOut, xMaxOut, yMaxOut := outBBoxes.GetF32(0), outBBoxes.GetF32(1), outBBoxes.GetF32(2), outBBoxes.GetF32(3)
		if math.Abs(float64(xMinOut-x)) <= 2 &&
			math.Abs(float64(yMinOut-y)) <= 2 &&
			math.Abs(float64(xMaxOut-x2)) <= 2 &&
			math.Abs(float64(yMaxOut-y2)) <= 2 {
			outKeypoint = keyPoint
			break
		}
	}
	return outBBoxes, outKeypoint, nil
}

func (c *FaceSelectionClient) getBiggestAreaFace(faceBoxes, faceKeyPoints *tensor.Dense) (*tensor.Dense, *tensor.Dense, error) {
	var biggestArea float32 = 0
	var biggestBbox, biggestKeyPoint *tensor.Dense

	for i := 0; i < faceBoxes.Shape()[0]; i++ {
		boxS, err := faceBoxes.Slice(tensor.S(i))
		if err != nil {
			return nil, nil, err
		}
		box := boxS.(*tensor.Dense)

		keyPointS, err := faceKeyPoints.Slice(tensor.S(i))
		if err != nil {
			return nil, nil, err
		}
		keyPoint := keyPointS.(*tensor.Dense)

		xMin, yMin, xMax, yMax := box.GetF32(0), box.GetF32(1), box.GetF32(2), box.GetF32(3)
		if (xMax-xMin)*(yMax-yMin) > biggestArea {
			biggestArea = (xMax - xMin) * (yMax - yMin)
			biggestBbox = box.Clone().(*tensor.Dense)
			biggestKeyPoint = keyPoint.Clone().(*tensor.Dense)
		}
	}
	return biggestBbox, biggestKeyPoint, nil
}

func (c *FaceSelectionClient) isFaceAreaBigEnough(img gocv.Mat, faceBox *tensor.Dense) bool {
	xMin, xMax := faceBox.GetF32(0), faceBox.GetF32(2)
	imgWidth := img.Size()[1]
	faceWidth := xMax - xMin
	return faceWidth/float32(imgWidth) > 0.25
}
