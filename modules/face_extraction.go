package modules

import (
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

type FaceExtractionClient struct {
	tritonClient *gotritonclient.TritonGRPCClient
	ModelParams  *config.ArcFaceRecognitionParams
	ModelConfig  *triton_proto.ModelConfigResponse
	imageSize    [2]int
	timeout      time.Duration
	batchSize    int
}

func NewFaceExtractionClient(tritonClient *gotritonclient.TritonGRPCClient, cfg *config.ArcFaceRecognitionParams) (*FaceExtractionClient, error) {
	client := &FaceExtractionClient{}
	client.ModelParams = cfg

	inferenceConfig, err := tritonClient.GetModelConfiguration(cfg.Timeout, cfg.ModelName, "")
	if err != nil {
		return nil, err
	}
	client.tritonClient = tritonClient
	client.ModelConfig = inferenceConfig
	client.imageSize = cfg.ImageSize
	client.batchSize = cfg.BatchSize

	return client, nil
}

func (c *FaceExtractionClient) Infer(imgs []gocv.Mat) ([]*tensor.Dense, error) {
	preprocessedImages, err := c.preprocess(imgs)
	if err != nil {
		return nil, err
	}

	outputs := make([][]*tensor.Dense, 0)

	for i := 0; i < preprocessedImages.Shape()[0]; i += c.batchSize {
		batch, err := preprocessedImages.Slice(tensor.S(i, i+c.batchSize), nil, nil, nil)
		if err != nil {
			return nil, err
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
					Fp32Contents: batch.(*tensor.Dense).Float32s(),
				},
			}
			modelInputs = append(modelInputs, modelInput)
		}

		modelRequest.Inputs = modelInputs
		inferResp, err := c.tritonClient.ModelGRPCInfer(c.ModelParams.Timeout, modelRequest)
		if err != nil {
			return nil, err
		}

		netOut := make([]*tensor.Dense, 0)
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
			netOut = append(netOut, outTensors)
		}
		outputs = append(outputs, netOut)
	}
	normalizedOutputs := make([]*tensor.Dense, 0)
	for i, o := range outputs {
		slice, err := o[0].Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}

		norm, err := utils.L2Norm(slice.(*tensor.Dense))
		if err != nil {
			return nil, err
		}

		apply, err := slice.(*tensor.Dense).Apply(func(x float32) float32 {
			return x / float32(norm)
		})
		if err != nil {
			return nil, err
		}
		normalizedOutputs = append(normalizedOutputs, apply.(*tensor.Dense))
	}

	return normalizedOutputs, nil
}

func (c *FaceExtractionClient) preprocess(imgs []gocv.Mat) (*tensor.Dense, error) {
	batchInputSize := int(math.Ceil(math.Max(math.Ceil(float64(len(imgs)/c.batchSize)), 1) * float64(c.batchSize)))

	preprocessedImages := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(batchInputSize, 3, c.imageSize[1], c.imageSize[0]),
	)

	for i, img := range imgs {
		resizedImg := gocv.NewMat()
		gocv.Resize(img, &resizedImg, image.Point{X: c.imageSize[0], Y: c.imageSize[1]}, 0, 0, gocv.InterpolationLinear)
		rgbImg := gocv.NewMat()
		gocv.CvtColor(resizedImg, &rgbImg, gocv.ColorBGRToRGB)
		_ = resizedImg.Close()

		imgShape := rgbImg.Size()
		imgTensors := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(imgShape[0], imgShape[1], 3),
		)

		for z := range 3 {
			for y := range imgShape[0] {
				for x := range imgShape[1] {
					err := imgTensors.SetAt((float32(rgbImg.GetVecbAt(y, x)[z])-127.5)*0.0078125, y, x, z)
					if err != nil {
						return nil, err
					}
				}
			}
		}
		_ = rgbImg.Close()

		err := imgTensors.T(2, 0, 1)
		if err != nil {
			return nil, err
		}
		preprocessedSlice, err := preprocessedImages.Slice(tensor.S(i))
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
