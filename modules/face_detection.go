package modules

import (
	"fmt"
	"github.com/elliotchance/orderedmap/v2"
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/processing"
	"github.com/okieraised/go-faceid-pipeline/rcnn"
	"github.com/okieraised/go-faceid-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/okieraised/go-triton-client/triton_proto"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
	"math"
)

type FaceDetectionClient struct {
	tritonClient        *gotritonclient.TritonGRPCClient
	ModelParams         *config.RetinaFaceDetectionParams
	ModelConfig         *triton_proto.ModelConfigResponse
	imageSize           [2]int
	useLandmarks        bool
	confidenceThreshold float32
	iouThreshold        float32
	fpnKeys             []string
	featStrideFPN       []int
	anchorConfig        *orderedmap.OrderedMap[string, processing.AnchorConfig]
	anchorsFPN          *orderedmap.OrderedMap[string, *tensor.Dense]
	numAnchors          *orderedmap.OrderedMap[string, int]
	pixelMeans          []float32
	pixelStds           []float32
	pixelScale          float32
	bboxStds            []float32
	landmarksStd        float32
}

func NewFaceDetectionClient(tritonClient *gotritonclient.TritonGRPCClient, cfg *config.RetinaFaceDetectionParams) (*FaceDetectionClient, error) {

	client := &FaceDetectionClient{}
	client.ModelParams = cfg

	inferenceConfig, err := tritonClient.GetModelConfiguration(cfg.Timeout, cfg.ModelName, "")
	if err != nil {
		return nil, err
	}
	client.tritonClient = tritonClient
	client.ModelConfig = inferenceConfig
	client.imageSize = cfg.ImageSize
	client.useLandmarks = true
	client.confidenceThreshold = cfg.ConfidenceThreshold
	client.iouThreshold = cfg.IOUThreshold

	client.featStrideFPN = []int{32, 16, 8}
	ratio := []float32{1.0}

	anchorConfig := orderedmap.NewOrderedMap[string, processing.AnchorConfig]()
	anchorConfig.Set(
		"32", processing.AnchorConfig{
			BaseSize:      16,
			Ratios:        ratio,
			Scales:        []float32{32, 16},
			AllowedBorder: 9999,
		},
	)
	anchorConfig.Set(
		"16", processing.AnchorConfig{
			BaseSize:      16,
			Ratios:        ratio,
			Scales:        []float32{8, 4},
			AllowedBorder: 9999,
		},
	)
	anchorConfig.Set(
		"8", processing.AnchorConfig{
			BaseSize:      16,
			Ratios:        ratio,
			Scales:        []float32{2, 1},
			AllowedBorder: 9999,
		},
	)
	client.anchorConfig = anchorConfig

	client.fpnKeys = make([]string, 0)

	for s, _ := range client.anchorConfig.Iterator() {
		client.fpnKeys = append(client.fpnKeys, fmt.Sprintf("stride%s", s))
	}

	anchorsFPN := orderedmap.NewOrderedMap[string, *tensor.Dense]()

	fpn, err := processing.GenerateAnchorsFPN2(false, client.anchorConfig)
	if err != nil {
		return nil, err
	}

	for idx, _ := range client.fpnKeys {
		anchorsFPN.Set(client.fpnKeys[idx], fpn[idx])
	}
	client.anchorsFPN = anchorsFPN

	numAnchors := orderedmap.NewOrderedMap[string, int]()
	anchorShape := make([]int, 0)

	for _, v := range client.anchorsFPN.Iterator() {
		anchorShape = append(anchorShape, v.Shape()[0])
	}

	for idx := range client.anchorsFPN.Keys() {
		numAnchors.Set(client.fpnKeys[idx], anchorShape[idx])
	}

	client.numAnchors = numAnchors

	client.pixelMeans = []float32{0, 0, 0}
	client.pixelStds = []float32{1, 1, 1}
	client.pixelScale = 1.0
	client.bboxStds = []float32{1, 1, 1, 1}
	client.landmarksStd = 1.0

	return client, nil
}

func (c *FaceDetectionClient) preprocess(img gocv.Mat) (gocv.Mat, float64, error) {

	imgShape := img.Size()
	imRatio := float64(imgShape[0]) / float64(imgShape[1])
	modelRatio := float64(c.imageSize[1]) / float64(c.imageSize[0])

	var newWidth, newHeight int

	if imRatio > modelRatio {
		newHeight = c.imageSize[1]
		newWidth = int(float64(newHeight) / imRatio)
	} else {
		newWidth = c.imageSize[0]
		newHeight = int(float64(newWidth) * imRatio)
	}
	detScale := float64(newHeight) / float64(imgShape[0])

	resizedImg := gocv.NewMat()
	defer resizedImg.Close()
	gocv.Resize(img, &resizedImg, image.Point{X: newWidth, Y: newHeight}, 0.0, 0.0, gocv.InterpolationLinear)

	detImg := gocv.NewMatWithSizesWithScalar([]int{c.imageSize[1], c.imageSize[0]}, gocv.MatTypeCV8UC3, gocv.NewScalar(0, 0, 0, 0))
	roi := detImg.Region(image.Rect(0, 0, newWidth, newHeight))
	gocv.Resize(resizedImg, &roi, image.Point{X: roi.Size()[1], Y: roi.Size()[0]}, 0, 0, gocv.InterpolationLinear)

	return detImg, detScale, nil
}

func (c *FaceDetectionClient) Infer(img gocv.Mat) (*tensor.Dense, *tensor.Dense, error) {

	proposalsList := make([]*tensor.Dense, 0)
	scoresList := make([]*tensor.Dense, 0)
	landmarksList := make([]*tensor.Dense, 0)

	preprocessedImg, preprocessedParam, err := c.preprocess(img)
	if err != nil {
		return nil, nil, err
	}

	imgShape := preprocessedImg.Size()
	imgTensors := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(1, 3, imgShape[0], imgShape[1]),
	)
	imgInfo := []int{imgShape[0], imgShape[1]}

	for z := range 3 {
		for y := range imgShape[0] {
			for x := range imgShape[1] {
				err := imgTensors.SetAt((float32(preprocessedImg.GetVecbAt(y, x)[2-z])/c.pixelScale-c.pixelMeans[2-z])/c.pixelStds[2-z], 0, z, y, x)
				if err != nil {
					return nil, nil, err
				}
			}
		}
	}

	modelRequest := &triton_proto.ModelInferRequest{
		ModelName: c.ModelParams.ModelName,
	}

	modelInputs := make([]*triton_proto.ModelInferRequest_InferInputTensor, 0)
	for _, inputCfg := range c.ModelConfig.Config.Input {
		modelInput := &triton_proto.ModelInferRequest_InferInputTensor{
			Name:     inputCfg.Name,
			Datatype: inputCfg.DataType.String()[5:],
			Shape:    inputCfg.Dims,
			Contents: &triton_proto.InferTensorContents{
				Fp32Contents: imgTensors.Float32s(),
			},
		}
		modelInputs = append(modelInputs, modelInput)
	}

	cfgOutputs := make([]*triton_proto.ModelInferRequest_InferRequestedOutputTensor, len(c.ModelConfig.Config.Output))
	for idx, outCfg := range c.ModelConfig.Config.Output {
		cfgOutputs[idx] = &triton_proto.ModelInferRequest_InferRequestedOutputTensor{
			Name: outCfg.Name,
		}
	}

	modelRequest.Inputs = modelInputs
	inferResp, err := c.tritonClient.ModelGRPCInfer(c.ModelParams.Timeout, modelRequest)
	if err != nil {
		return nil, nil, err
	}
	netOut := make([]*tensor.Dense, len(cfgOutputs))
	for idx, out := range inferResp.Outputs {
		outShape := make([]int, 0)
		for _, shape := range out.Shape {
			outShape = append(outShape, int(shape))
		}
		outTensors := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(outShape...),
			tensor.WithBacking(utils.BytesToT32[float32](inferResp.RawOutputContents[idx])),
		)

		for subIdx, cfg := range cfgOutputs {
			if out.Name == cfg.Name {
				netOut[subIdx] = outTensors
			}
		}
	}

	symIdx := 0
	for _, s := range c.featStrideFPN {
		anchorIdx, _ := c.numAnchors.Get(fmt.Sprintf("stride%d", s))
		scores, err := netOut[symIdx].Slice(nil, tensor.S(anchorIdx, netOut[symIdx].Shape()[2]), nil, nil)
		if err != nil {
			return nil, nil, err
		}
		bboxDeltas := netOut[symIdx+1]
		height, width := bboxDeltas.Shape()[2], bboxDeltas.Shape()[3]
		A, _ := c.numAnchors.Get(fmt.Sprintf("stride%d", s))
		K := height * width
		anchorsFPN, _ := c.anchorsFPN.Get(fmt.Sprintf("stride%d", s))
		anchors, err := rcnn.Anchors(height, width, s, anchorsFPN)
		if err != nil {
			return nil, nil, err
		}

		err = anchors.Reshape(K*A, 4)
		if err != nil {
			return nil, nil, err
		}

		err = scores.T(0, 2, 3, 1)
		if err != nil {
			return nil, nil, err
		}

		resizedScores := scores.Clone()
		err = resizedScores.(*tensor.Dense).Reshape(scores.DataSize(), 1)
		if err != nil {
			return nil, nil, err
		}

		err = bboxDeltas.T(0, 2, 3, 1)
		if err != nil {
			return nil, nil, err
		}

		bboxPredLen := int(math.Floor(float64(bboxDeltas.Shape()[3] / A)))
		err = bboxDeltas.Reshape(bboxDeltas.DataSize()/bboxPredLen, bboxPredLen)
		if err != nil {
			return nil, nil, err
		}

		for i := 0; i < 4; i++ {
			slice, err := bboxDeltas.Slice(nil, tensor.S(i, 4))
			if err != nil {
				return nil, nil, err
			}
			scaled, err := slice.(*tensor.Dense).MulScalar(c.bboxStds[i], true)
			if err != nil {
				return nil, nil, err
			}
			err = tensor.Copy(slice, scaled)
			if err != nil {
				return nil, nil, err
			}
		}
		proposals, err := c.bboxPred(anchors, bboxDeltas)
		if err != nil {
			return nil, nil, err
		}
		proposals, err = processing.ClipBoxes(proposals, imgInfo)
		if err != nil {
			return nil, nil, err
		}

		scoreRavel := resizedScores.(*tensor.Dense).Clone().(*tensor.Dense)
		err = scoreRavel.Reshape(scoreRavel.Shape()[0])
		if err != nil {
			return nil, nil, err
		}

		order := make([]int, 0)
		for i, v := range scoreRavel.Float32s() {
			if v >= c.confidenceThreshold {
				order = append(order, i)
			}
		}

		proposals, err = utils.SelectRows2D(proposals, order)
		if err != nil {
			return nil, nil, err
		}

		scores, err = utils.SelectRows2D(resizedScores.(*tensor.Dense), order)
		if err != nil {
			return nil, nil, err
		}

		proposalsList = append(proposalsList, proposals)
		scoresList = append(scoresList, scores.(*tensor.Dense))

		if c.useLandmarks {
			landmarkDeltas := netOut[symIdx+2]
			landmarkPredLen := float32(math.Floor(float64(landmarkDeltas.Shape()[1]) / float64(A)))

			err = landmarkDeltas.T(0, 2, 3, 1)
			if err != nil {
				return nil, nil, err
			}

			err = landmarkDeltas.Reshape(
				landmarkDeltas.DataSize()/(5*int(math.Floor(float64(landmarkPredLen)/float64(5)))),
				5,
				int(math.Floor(float64(landmarkPredLen)/float64(5))),
			)
			if err != nil {
				return nil, nil, err
			}

			landmarkDeltas, err = landmarkDeltas.MulScalar(c.landmarksStd, true)
			if err != nil {
				return nil, nil, err
			}

			landmarks, err := c.landmarkPred(anchors, landmarkDeltas)
			if err != nil {
				return nil, nil, err
			}
			landmarks, err = utils.SelectRows3D(landmarks, order)
			if err != nil {
				return nil, nil, err
			}
			landmarksList = append(landmarksList, landmarks)
		}

		if c.useLandmarks {
			symIdx += 3
		} else {
			symIdx += 2
		}
	}

	proposals, err := utils.VStack(proposalsList)
	if err != nil {
		return nil, nil, err
	}

	var landmarks, det *tensor.Dense
	if proposals.Shape()[0] == 0 {
		if c.useLandmarks {
			landmarks = tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(0, 5, 2),
			)
			det = tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(0, 5),
			)
		}
		return det, landmarks, nil
	}

	scores, err := utils.VStack(scoresList)
	if err != nil {
		return nil, nil, err
	}

	scoresRavel := scores.Clone().(*tensor.Dense)
	err = scoresRavel.Reshape(scoresRavel.Shape()[0])
	if err != nil {
		return nil, nil, err
	}

	order, err := utils.ArgSortDescending(scoresRavel)
	if err != nil {
		return nil, nil, err
	}

	proposals, err = utils.SelectRows2D(proposals, order)
	if err != nil {
		return nil, nil, err
	}

	scores, err = utils.SelectRows2D(scores, order)
	if err != nil {
		return nil, nil, err
	}

	if c.useLandmarks {
		landmarks, err = utils.VStack(landmarksList)
		if err != nil {
			return nil, nil, err
		}
		landmarks, err = utils.SelectRows3D(landmarks, order)
		if err != nil {
			return nil, nil, err
		}
	}

	proposalSlice, err := proposals.Slice(nil, tensor.S(0, 4, 1))
	if err != nil {
		return nil, nil, err
	}
	preDet, err := utils.HStack([]*tensor.Dense{proposalSlice.(*tensor.Dense), scores})
	if err != nil {
		return nil, nil, err
	}

	keep, err := processing.NMS(preDet, c.iouThreshold)
	if err != nil {
		return nil, nil, err
	}

	if proposals.Shape()[1] > 4 {
		pSliceRemaining, err := proposals.Slice(nil, tensor.S(4, proposals.Shape()[1]))
		if err != nil {
			return nil, nil, err
		}
		det, err = preDet.Hstack(pSliceRemaining.(*tensor.Dense))
		if err != nil {
			return nil, nil, err
		}
	} else {
		det = preDet
	}

	det, err = utils.SelectRows2D(det, keep)
	if err != nil {
		return nil, nil, err
	}

	if c.useLandmarks {
		landmarks, err = utils.SelectRows3D(landmarks, keep)
		if err != nil {
			return nil, nil, err
		}
	}
	return c.postprocess(det, landmarks, preprocessedParam)
}

func (c *FaceDetectionClient) postprocess(det, landmark *tensor.Dense, preprocessedParam float64) (*tensor.Dense, *tensor.Dense, error) {
	detSlice, err := det.Slice(nil, tensor.S(0, 4))
	if err != nil {
		return nil, nil, err
	}

	detSliceOwned := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(detSlice.Shape()...),
	)

	err = tensor.Copy(detSliceOwned, detSlice)
	if err != nil {
		return nil, nil, err
	}

	scaledDet, err := detSliceOwned.DivScalar(float32(preprocessedParam), true)
	if err != nil {
		return nil, nil, err
	}

	err = tensor.Copy(detSlice, scaledDet)
	if err != nil {
		return nil, nil, err
	}

	if landmark != nil {
		landmark, err = landmark.DivScalar(float32(preprocessedParam), true)
		if err != nil {
			return nil, nil, err
		}
	}

	return det, landmark, nil
}

func (c *FaceDetectionClient) landmarkPred(boxes, landmarkDeltas *tensor.Dense) (*tensor.Dense, error) {
	if boxes.Shape()[0] == 0 {
		return tensor.New(
			tensor.Of(tensor.Float32), tensor.WithShape(0, landmarkDeltas.Shape()[1]),
		), nil
	}

	boxes0, err := boxes.Slice(nil, tensor.S(0))
	if err != nil {
		return nil, err
	}
	boxes00 := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(boxes0.Shape()...),
	)
	err = tensor.Copy(boxes00, boxes0)
	if err != nil {
		return nil, err
	}

	boxes1, err := boxes.Slice(nil, tensor.S(1))
	if err != nil {
		return nil, err
	}
	boxes11 := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(boxes1.Shape()...),
	)
	err = tensor.Copy(boxes11, boxes1)
	if err != nil {
		return nil, err
	}

	boxes2, err := boxes.Slice(nil, tensor.S(2))
	if err != nil {
		return nil, err
	}
	boxes22 := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(boxes2.Shape()...),
	)
	err = tensor.Copy(boxes22, boxes2)
	if err != nil {
		return nil, err
	}

	boxes3, err := boxes.Slice(nil, tensor.S(3))
	if err != nil {
		return nil, err
	}
	boxes33 := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(boxes3.Shape()...),
	)
	err = tensor.Copy(boxes33, boxes3)
	if err != nil {
		return nil, err
	}

	// widths
	widths, err := boxes22.Sub(boxes00)
	if err != nil {
		return nil, err
	}
	widths, err = widths.AddScalar(float32(1.0), true)
	if err != nil {
		return nil, err
	}

	// heights
	heights, err := boxes33.Sub(boxes11)
	if err != nil {
		return nil, err
	}
	heights, err = heights.AddScalar(float32(1.0), true)
	if err != nil {
		return nil, err
	}

	// centerX
	scaledWidth, err := widths.Apply(func(x float32) float32 {
		return 0.5 * (x - 1)
	})

	centerX, err := boxes00.Add(scaledWidth.(*tensor.Dense))
	if err != nil {
		return nil, err
	}

	// centerY
	scaledHeight, err := heights.Apply(func(x float32) float32 {
		return 0.5 * (x - 1)
	})
	if err != nil {
		return nil, err
	}
	centerY, err := boxes11.Add(scaledHeight.(*tensor.Dense))
	if err != nil {
		return nil, err
	}

	pred := landmarkDeltas.Clone().(*tensor.Dense)

	for i := range 5 {
		lmkSlices0, err := landmarkDeltas.Slice(nil, tensor.S(i), tensor.S(0))
		if err != nil {
			return nil, err
		}

		scaledLmk0, err := lmkSlices0.(*tensor.Dense).Mul(widths)
		if err != nil {
			return nil, err
		}
		newPred0, err := scaledLmk0.Add(centerX)
		if err != nil {
			return nil, err
		}

		predSlice0, err := pred.Slice(nil, tensor.S(i), tensor.S(0))
		if err != nil {
			return nil, err
		}

		err = tensor.Copy(predSlice0, newPred0)
		if err != nil {
			return nil, err
		}

		lmkSlices1, err := landmarkDeltas.Slice(nil, tensor.S(i), tensor.S(1))
		if err != nil {
			return nil, err
		}

		scaledLmk1, err := lmkSlices1.(*tensor.Dense).Mul(heights)
		if err != nil {
			return nil, err
		}
		newPred1, err := scaledLmk1.Add(centerY)
		if err != nil {
			return nil, err
		}

		predSlice1, err := pred.Slice(nil, tensor.S(i), tensor.S(1))
		if err != nil {
			return nil, err
		}

		err = tensor.Copy(predSlice1, newPred1)
		if err != nil {
			return nil, err
		}
	}

	return pred, nil
}

func (c *FaceDetectionClient) bboxPred(boxes, bboxDelta *tensor.Dense) (*tensor.Dense, error) {
	if boxes.Shape()[0] == 0 {
		return tensor.New(
			tensor.Of(tensor.Float32), tensor.WithShape(0, bboxDelta.Shape()[1]),
		), nil
	}

	boxes0, err := boxes.Slice(nil, tensor.S(0))
	if err != nil {
		return nil, err
	}
	boxes00 := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(boxes0.Shape()...),
	)
	err = tensor.Copy(boxes00, boxes0)
	if err != nil {
		return nil, err
	}

	boxes1, err := boxes.Slice(nil, tensor.S(1))
	if err != nil {
		return nil, err
	}
	boxes11 := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(boxes1.Shape()...),
	)
	err = tensor.Copy(boxes11, boxes1)
	if err != nil {
		return nil, err
	}

	boxes2, err := boxes.Slice(nil, tensor.S(2))
	if err != nil {
		return nil, err
	}
	boxes22 := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(boxes2.Shape()...),
	)
	err = tensor.Copy(boxes22, boxes2)
	if err != nil {
		return nil, err
	}

	boxes3, err := boxes.Slice(nil, tensor.S(3))
	if err != nil {
		return nil, err
	}
	boxes33 := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(boxes3.Shape()...),
	)
	err = tensor.Copy(boxes33, boxes3)
	if err != nil {
		return nil, err
	}

	// widths
	widths, err := boxes22.Sub(boxes00)
	if err != nil {
		return nil, err
	}
	widths, err = widths.AddScalar(float32(1.0), true)
	if err != nil {
		return nil, err
	}

	// heights
	heights, err := boxes33.Sub(boxes11)
	if err != nil {
		return nil, err
	}
	heights, err = heights.AddScalar(float32(1.0), true)
	if err != nil {
		return nil, err
	}

	// centerX
	scaledWidth, err := widths.Apply(func(x float32) float32 {
		return 0.5 * (x - 1)
	})

	centerX, err := boxes00.Add(scaledWidth.(*tensor.Dense))
	if err != nil {
		return nil, err
	}

	// centerY
	scaledHeight, err := heights.Apply(func(x float32) float32 {
		return 0.5 * (x - 1)
	})
	if err != nil {
		return nil, err
	}
	centerY, err := boxes11.Add(scaledHeight.(*tensor.Dense))
	if err != nil {
		return nil, err
	}

	dx, err := bboxDelta.Slice(nil, tensor.S(0, 1))
	if err != nil {
		return nil, err
	}
	dy, err := bboxDelta.Slice(nil, tensor.S(1, 2))
	if err != nil {
		return nil, err
	}
	dw, err := bboxDelta.Slice(nil, tensor.S(2, 3))
	if err != nil {
		return nil, err
	}
	dh, err := bboxDelta.Slice(nil, tensor.S(3, 4))
	if err != nil {
		return nil, err
	}

	newWidthShape := []int{widths.Shape()[0], 1}
	err = widths.Reshape(newWidthShape...)
	if err != nil {
		return nil, err
	}

	newHeightShape := []int{heights.Shape()[0], 1}
	err = heights.Reshape(newHeightShape...)
	if err != nil {
		return nil, err
	}

	newCenterXShape := []int{centerX.Shape()[0], 1}
	err = centerX.Reshape(newCenterXShape...)
	if err != nil {
		return nil, err
	}

	newCenterYShape := []int{centerY.Shape()[0], 1}
	err = centerY.Reshape(newCenterYShape...)
	if err != nil {
		return nil, err
	}

	scaledWidths, err := widths.Mul(dx.(*tensor.Dense))
	if err != nil {
		return nil, err
	}
	predCenterX, err := scaledWidths.Add(centerX)
	if err != nil {
		return nil, err
	}

	scaledHeighs, err := heights.Mul(dy.(*tensor.Dense))
	if err != nil {
		return nil, err
	}
	predCenterY, err := scaledHeighs.Add(centerY)
	if err != nil {
		return nil, err
	}

	expDw, err := dw.(*tensor.Dense).Apply(func(x float32) float32 {
		return float32(math.Exp(float64(x)))
	})
	if err != nil {
		return nil, err
	}

	expDh, err := dh.(*tensor.Dense).Apply(func(x float32) float32 {
		return float32(math.Exp(float64(x)))
	})
	if err != nil {
		return nil, err
	}

	predW, err := expDw.(*tensor.Dense).Mul(widths)
	if err != nil {
		return nil, err
	}

	predH, err := expDh.(*tensor.Dense).Mul(heights)
	if err != nil {
		return nil, err
	}

	predBoxes := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(bboxDelta.Shape()...),
	)

	predBox01, err := predBoxes.Slice(nil, tensor.S(0))
	if err != nil {
		return nil, err
	}

	subPredW, err := predW.Apply(func(x float32) float32 {
		return (x - 1) * 0.5
	})
	if err != nil {
		return nil, err
	}

	subPredH, err := predH.Apply(func(x float32) float32 {
		return (x - 1) * 0.5
	})
	if err != nil {
		return nil, err
	}

	newPredBox01, err := predCenterX.Sub(subPredW.(*tensor.Dense))
	if err != nil {
		return nil, err
	}

	err = tensor.Copy(predBox01, newPredBox01)
	if err != nil {
		return nil, err
	}

	predBox12, err := predBoxes.Slice(nil, tensor.S(1))
	if err != nil {
		return nil, err
	}
	newPredBox12, err := predCenterY.Sub(subPredH.(*tensor.Dense))
	if err != nil {
		return nil, err
	}

	err = tensor.Copy(predBox12, newPredBox12)
	if err != nil {
		return nil, err
	}

	predBox23, err := predBoxes.Slice(nil, tensor.S(2))
	if err != nil {
		return nil, err
	}
	newPredBox23, err := predCenterX.Add(subPredW.(*tensor.Dense))
	if err != nil {
		return nil, err
	}

	err = tensor.Copy(predBox23, newPredBox23)
	if err != nil {
		return nil, err
	}

	predBox34, err := predBoxes.Slice(nil, tensor.S(3))
	if err != nil {
		return nil, err
	}

	newPredBox34, err := predCenterY.Add(subPredH.(*tensor.Dense))
	if err != nil {
		return nil, err
	}

	err = tensor.Copy(predBox34, newPredBox34)
	if err != nil {
		return nil, err
	}

	if bboxDelta.Shape()[1] > 4 {
		slicedPredBoxes, err := predBoxes.Slice(nil, tensor.S(4, predBoxes.Shape()[1]-1))
		if err != nil {
			return nil, err
		}

		slicedBBoxDelta, err := bboxDelta.Slice(nil, tensor.S(4, bboxDelta.Shape()[1]-1))
		if err != nil {
			return nil, err
		}
		err = tensor.Copy(slicedPredBoxes, slicedBBoxDelta)
		if err != nil {
			return nil, err
		}
	}

	return predBoxes, nil
}
