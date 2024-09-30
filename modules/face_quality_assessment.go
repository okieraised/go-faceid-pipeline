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

type FaceQualityAssessmentClient struct {
	tritonClient *gotritonclient.TritonGRPCClient
	ModelParams  *config.FaceQualityAssessmentParams
	ModelConfig  *triton_proto.ModelConfigResponse
	timeout      time.Duration
	modelName    string
	threshold    float32
	imageSize    [2]int
	batchSize    int
}

func NewFaceQualityAssessmentClient(tritonClient *gotritonclient.TritonGRPCClient, cfg *config.FaceQualityAssessmentParams) (*FaceQualityAssessmentClient, error) {
	client := &FaceQualityAssessmentClient{}

	inferenceConfig, err := tritonClient.GetModelConfiguration(cfg.Timeout, cfg.ModelName, "")
	if err != nil {
		return nil, err
	}

	client.ModelParams = cfg
	client.tritonClient = tritonClient
	client.ModelConfig = inferenceConfig
	client.timeout = cfg.Timeout
	client.modelName = cfg.ModelName
	client.threshold = cfg.Threshold
	client.imageSize = cfg.ImageSize
	client.batchSize = cfg.BatchSize

	return client, nil
}

func (c *FaceQualityAssessmentClient) Infer(imgs []gocv.Mat) ([]float32, []int, error) {

	idxs := make([]int, 0)
	scores := make([]float32, 0)
	for idx := range len(imgs) {
		imgTensors, err := c.preprocess(imgs[idx])
		if err != nil {
			return nil, nil, err
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
			return nil, nil, err
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
		score, err := outTensors.At(0, 0)
		if err != nil {
			return nil, nil, err
		}
		var predict int
		if score.(float32) > c.threshold {
			predict = 1
		}
		idxs = append(idxs, predict)
		scores = append(scores, score.(float32))
	}
	return scores, idxs, nil
}

func (c *FaceQualityAssessmentClient) preprocess(img gocv.Mat) (*tensor.Dense, error) {
	var err error

	resizedImg := gocv.NewMat()
	defer func() {
		cErr := resizedImg.Close()
		if cErr != nil && err == nil {
			err = cErr
		}
	}()

	gocv.Resize(img, &resizedImg, image.Point{X: c.imageSize[0], Y: c.imageSize[1]}, 0, 0, gocv.InterpolationLinear)
	rgbImg := gocv.NewMat()
	gocv.CvtColor(resizedImg, &rgbImg, gocv.ColorBGRToRGB)
	defer func() {
		cErr := rgbImg.Close()
		if cErr != nil && err == nil {
			err = cErr
		}
	}()

	imgShape := rgbImg.Size()
	imgTensors := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(1, 3, imgShape[0], imgShape[1]),
	)

	for z := range 3 {
		for y := range imgShape[0] {
			for x := range imgShape[1] {
				err = imgTensors.SetAt((float32(rgbImg.GetVecbAt(y, x)[z])-127.5)*0.00784313725, 0, z, y, x)
				if err != nil {
					return nil, err
				}
			}
		}
	}

	err = imgTensors.T(0, 3, 2, 1)
	if err != nil {
		return nil, err
	}
	return imgTensors, nil
}
