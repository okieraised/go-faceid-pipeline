package modules

import (
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/okieraised/go-triton-client/triton_proto"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
	"time"
)

type FaceQualityClient struct {
	tritonClient *gotritonclient.TritonGRPCClient
	ModelParams  *config.FaceQualityParams
	ModelConfig  *triton_proto.ModelConfigResponse
	imageSize    [2]int
	timeout      time.Duration
	batchSize    int
	threshold    float32
}

func NewFaceQualityClient(tritonClient *gotritonclient.TritonGRPCClient, cfg *config.FaceQualityParams) (*FaceQualityClient, error) {
	client := &FaceQualityClient{}
	client.ModelParams = cfg

	inferenceConfig, err := tritonClient.GetModelConfiguration(cfg.Timeout, cfg.ModelName, "")
	if err != nil {
		return nil, err
	}
	client.tritonClient = tritonClient
	client.ModelConfig = inferenceConfig
	client.imageSize = cfg.ImageSize
	client.batchSize = cfg.BatchSize
	client.threshold = cfg.Threshold

	return client, nil
}

func (c *FaceQualityClient) Infer(imgs []gocv.Mat) ([]float32, []int, error) {

	batchSize := len(imgs)
	scores := make([]float32, 0)
	idxs := make([]int, 0)

	means := []float32{123.675, 116.28, 103.53}
	std := []float32{0.01712475, 0.017507, 0.01742919}

	for i := range batchSize {
		resizedImg := gocv.NewMat()
		gocv.Resize(imgs[i], &resizedImg, image.Point{X: c.imageSize[0], Y: c.imageSize[1]}, 0, 0, gocv.InterpolationLinear)
		rgbImg := gocv.NewMat()
		gocv.CvtColor(resizedImg, &rgbImg, gocv.ColorBGRToRGB)
		_ = resizedImg.Close()

		imgShape := rgbImg.Size()
		imgTensors := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(1, 3, imgShape[0], imgShape[1]),
		)

		for z := range 3 {
			for y := range imgShape[0] {
				for x := range imgShape[1] {
					err := imgTensors.SetAt((float32(rgbImg.GetVecbAt(y, x)[z])-means[z])*std[z], 0, z, y, x)
					if err != nil {
						return scores, idxs, err
					}
				}
			}
		}
		_ = rgbImg.Close()

		err := imgTensors.T(0, 3, 1, 2)
		if err != nil {
			return scores, idxs, err
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

		modelRequest.Inputs = modelInputs
		inferResp, err := c.tritonClient.ModelGRPCInfer(c.ModelParams.Timeout, modelRequest)
		if err != nil {
			return scores, idxs, err
		}

		outShape := make([]int, 0)
		for _, shape := range inferResp.Outputs[0].Shape {
			outShape = append(outShape, int(shape))
		}

		outTensors := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(outShape...),
			tensor.WithBacking(utils.BytesToT32[float32](inferResp.RawOutputContents[0])),
		)

		predict, err := utils.ArgMax(outTensors)
		if err != nil {
			return scores, idxs, err
		}
		score := outTensors.Float32s()[predict]
		if predict == 1 && score < c.threshold {
			predict = 0
			score = outTensors.Float32s()[predict]
		}

		idxs = append(idxs, predict)
		scores = append(scores, score)
	}

	return scores, idxs, nil
}
