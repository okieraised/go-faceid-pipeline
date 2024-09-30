package modules

import (
	"errors"
	"fmt"
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/okieraised/go-triton-client/triton_proto"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
	"math"
	"time"
)

type FaceAntiSpoofingClient struct {
	tritonClient *gotritonclient.TritonGRPCClient
	ModelParams  *config.FaceAntiSpoofingParam
	timeout      time.Duration
	modelNames   []string
	scales       []float32
	threshold    float32
	imageSize    [][2]int
	batchSize    int
}

type scaleParam struct {
	orgImg gocv.Mat
	bbox   []int
	scale  float32
	outW   int
	outH   int
	crop   bool
}

func NewFaceAntiSpoofingClient(tritonClient *gotritonclient.TritonGRPCClient, cfg *config.FaceAntiSpoofingParam) *FaceAntiSpoofingClient {
	client := &FaceAntiSpoofingClient{}
	client.ModelParams = cfg
	client.tritonClient = tritonClient
	client.timeout = cfg.Timeout
	client.modelNames = cfg.ModelNames
	client.scales = cfg.Scales
	client.threshold = cfg.Threshold
	client.imageSize = cfg.ImageSizes
	client.batchSize = cfg.BatchSize

	return client
}

func (c *FaceAntiSpoofingClient) Infer(imgs []gocv.Mat, faceBoxes []*tensor.Dense) ([]*tensor.Dense, error) {

	listImageScales := make([][]gocv.Mat, len(c.scales))
	listWeightScales := make([][]float64, len(c.scales))

	if len(imgs) != len(faceBoxes) {
		return nil, errors.New("number of images and face boxes must be equal")
	}

	for idx := range len(imgs) {
		bgrImg := gocv.NewMat()
		gocv.CvtColor(imgs[idx], &bgrImg, gocv.ColorRGBToBGR)
		tmps, weights, err := c.getScaleImage(bgrImg, faceBoxes[idx])
		if err != nil {
			return nil, err
		}
		for i := range c.scales {
			listImageScales[i] = append(listImageScales[i], tmps[i])
			listWeightScales[i] = append(listWeightScales[i], weights[i])
		}
	}

	outputs := make([][]*tensor.Dense, 0)
	for idx := range c.scales {
		preprocessedImages, err := c.preprocess(listImageScales[idx], idx)
		if err != nil {
			return nil, err
		}

		for i := 0; i < preprocessedImages.Shape()[0]; i += c.batchSize {
			tensorS, err := preprocessedImages.Slice(tensor.S(i, i+c.batchSize), nil, nil, nil)
			if err != nil {
				return nil, err
			}

			inputTensor := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(tensorS.(*tensor.Dense).Shape()...))
			err = tensor.Copy(inputTensor, tensorS)
			if err != nil {
				return nil, err
			}

			inferenceConfig, err := c.tritonClient.GetModelConfiguration(c.ModelParams.Timeout, c.ModelParams.ModelNames[idx], "")
			if err != nil {
				return nil, err
			}

			modelRequest := &triton_proto.ModelInferRequest{
				ModelName: c.ModelParams.ModelNames[idx],
			}

			modelInputs := make([]*triton_proto.ModelInferRequest_InferInputTensor, 0)
			modelInput := &triton_proto.ModelInferRequest_InferInputTensor{
				Name:     inferenceConfig.Config.Input[0].Name,
				Datatype: inferenceConfig.Config.Input[0].DataType.String()[5:],
				Shape:    inferenceConfig.Config.Input[0].Dims,
				Contents: &triton_proto.InferTensorContents{
					Fp32Contents: inputTensor.Float32s(),
				},
			}
			modelInputs = append(modelInputs, modelInput)

			modelRequest.Inputs = modelInputs
			inferResp, err := c.tritonClient.ModelGRPCInfer(c.ModelParams.Timeout, modelRequest)
			if err != nil {
				return nil, err
			}

			netOut := make([]*tensor.Dense, 0)
			for sIdx, out := range inferResp.Outputs {
				outShape := make([]int, 0)
				for _, shape := range out.Shape {
					outShape = append(outShape, int(shape))
				}
				outTensors := tensor.New(
					tensor.Of(tensor.Float32),
					tensor.WithShape(outShape...),
					tensor.WithBacking(utils.BytesToT32[float32](inferResp.RawOutputContents[sIdx])),
				)
				netOut = append(netOut, outTensors)
			}
			outputs = append(outputs, netOut)
		}
	}

	results, err := c.postprocess(outputs, listWeightScales)
	if err != nil {
		return nil, err
	}
	return results, nil
}

func (c *FaceAntiSpoofingClient) postprocess(outputs [][]*tensor.Dense, listWeightScales [][]float64) ([]*tensor.Dense, error) {
	score, err := c.liveScore(outputs, listWeightScales)
	if err != nil {
		return nil, err
	}

	return score, nil
}

func (c *FaceAntiSpoofingClient) preprocess(imgs []gocv.Mat, idx int) (*tensor.Dense, error) {
	batchInputSize := int(math.Ceil(math.Max(math.Ceil(float64(len(imgs)/c.batchSize)), 1) * float64(c.batchSize)))

	preprocessedImages := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(batchInputSize, 3, c.imageSize[idx][1], c.imageSize[idx][0]),
	)

	for i, img := range imgs {
		bgrImg := gocv.NewMat()
		gocv.CvtColor(img, &bgrImg, gocv.ColorRGBToBGR)
		_ = img.Close()

		imgShape := bgrImg.Size()
		imgTensors := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(imgShape[0], imgShape[1], 3),
		)

		for z := range 3 {
			for y := range imgShape[0] {
				for x := range imgShape[1] {
					err := imgTensors.SetAt(float32(bgrImg.GetVecbAt(y, x)[z]), y, x, z)
					if err != nil {
						return nil, err
					}
				}
			}
		}
		err := bgrImg.Close()
		if err != nil {
			return nil, err
		}

		err = imgTensors.T(2, 0, 1)
		if err != nil {
			return nil, err
		}

		preprocessedSlice, err := preprocessedImages.Slice(tensor.S(i), nil, nil, nil)
		if err != nil {
			return nil, err
		}

		err = tensor.Copy(preprocessedSlice, imgTensors)
		if err != nil {
			return nil, err
		}
	}

	return preprocessedImages, nil
}

func (c *FaceAntiSpoofingClient) getScaleImage(img gocv.Mat, faceBox *tensor.Dense) ([]gocv.Mat, []float64, error) {
	detXmin, detYmin, detXmax, detYmax := faceBox.GetF32(0), faceBox.GetF32(1), faceBox.GetF32(2), faceBox.GetF32(3)
	detHeight := detYmax - detYmin
	cX := (detXmin + detXmax) / 2
	left := int(cX - 0.47*detHeight)
	right := int(cX + 0.47*detHeight)
	top := detYmin
	bottom := detYmax
	bbox := []int{int(left), int(top), int(right - left + 1), int(bottom - top + 1)}

	crops := make([]gocv.Mat, 0)
	weights := make([]float64, 0)
	for i, scale := range c.scales {
		params := scaleParam{
			orgImg: img,
			bbox:   bbox,
			scale:  scale,
			outW:   c.imageSize[i][0],
			outH:   c.imageSize[i][1],
			crop:   true,
		}
		crop, weight, err := c.cropImage(params)
		if err != nil {
			return crops, weights, err
		}
		crops = append(crops, crop)
		weights = append(weights, weight)
	}

	return crops, weights, nil
}

func (c *FaceAntiSpoofingClient) cropImage(params scaleParam) (gocv.Mat, float64, error) {
	dstImg := gocv.NewMat()
	var weight float64
	var leftTopX, leftTopY, rightBottomX, rightBottomY int

	if !params.crop {
		gocv.Resize(params.orgImg, &dstImg, image.Point{X: params.outW, Y: params.outH}, 0, 0, gocv.InterpolationLinear)
	} else {
		srcShape := params.orgImg.Size()
		srcH, srcW := srcShape[0], srcShape[1]
		leftTopX, leftTopY, rightBottomX, rightBottomY, weight = c.getNewBox(srcW, srcH, params.bbox, params.scale)

		img := params.orgImg.Region(image.Rect(leftTopX, leftTopY, rightBottomX+1, rightBottomY+1))
		defer img.Close()

		gocv.Resize(img, &dstImg, image.Pt(params.outW, params.outH), 0, 0, gocv.InterpolationLinear)
	}
	return dstImg, weight, nil
}

func (c *FaceAntiSpoofingClient) getNewBox(srcW, srcH int, bbox []int, scaleOri float32) (int, int, int, int, float64) {
	x := bbox[0]
	y := bbox[1]
	boxW := bbox[2]
	boxH := bbox[3]

	scale := math.Min(float64(srcH-1)/float64(boxH), math.Min(float64(srcW-1)/float64(boxW), float64(scaleOri)))

	newWidth := float64(boxW) * scale
	newHeight := float64(boxH) * scale
	centerX, centerY := float64(boxW)/2+float64(x), float64(boxH)/2+float64(y)

	leftTopX := centerX - newWidth/2
	leftTopY := centerY - newHeight/2
	rightBottomX := centerX + newWidth/2
	rightBottomY := centerY + newHeight/2

	if leftTopX < 0 {
		rightBottomX -= leftTopX
		leftTopX = 0
	}

	if leftTopY < 0 {
		rightBottomY -= leftTopY
		leftTopY = 0
	}

	if rightBottomX > float64(srcW)-1 {
		leftTopX -= rightBottomX - float64(srcW) + 1
		rightBottomX = float64(srcW) - 1
	}

	if rightBottomY > float64(srcH)-1 {
		leftTopY -= rightBottomY - float64(srcH) + 1
		rightBottomY = float64(srcH) - 1
	}

	return int(leftTopX), int(leftTopY), int(rightBottomX), int(rightBottomY), scale / float64(scaleOri)
}

func (c *FaceAntiSpoofingClient) liveScore(outputs [][]*tensor.Dense, listWeightScales [][]float64) ([]*tensor.Dense, error) {
	results := make([]*tensor.Dense, 0)

	for i := 0; i < len(outputs[0]); i++ {
		var weightedSum *tensor.Dense
		weightsSum := 0.0
		for j := 0; j < len(outputs); j++ {
			output := outputs[j][i]
			weight := listWeightScales[j][i]
			weightsSum += weight
			oCol1, err := output.Slice(nil, tensor.S(1))
			if err != nil {
				return nil, err
			}
			err = oCol1.Reshape(1)
			if err != nil {
				return nil, err
			}
			weightedCol1, err := oCol1.(*tensor.Dense).MulScalar(float32(weight), false)
			if err != nil {
				return nil, err
			}
			if weightedSum == nil {
				weightedSum = weightedCol1.Clone().(*tensor.Dense)
			} else {
				weightedSum, err = weightedSum.Add(weightedCol1)
				if err != nil {
					return nil, err
				}
			}
		}

		if weightsSum == 0 {
			return nil, fmt.Errorf("sum of weights is zero")
		}
		result, err := weightedSum.DivScalar(float32(weightsSum), true)
		if err != nil {
			return nil, err
		}
		livenesses := make([]int, 0)
		for _, val := range result.Float32s() {
			var bitSetVar int
			if val > c.threshold {
				bitSetVar = 1
			}
			livenesses = append(livenesses, bitSetVar)
		}

		liveness := tensor.New(
			tensor.Of(tensor.Int),
			tensor.WithShape(result.Shape()...),
			tensor.WithBacking(livenesses),
		)

		results = append(results, liveness)
	}

	return results, nil
}
