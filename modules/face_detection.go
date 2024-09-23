package modules

import (
	"fmt"
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/processing"
	"github.com/okieraised/go-faceid-pipeline/rcnn"
	"github.com/okieraised/go-faceid-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/okieraised/go-triton-client/triton_proto"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
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
	anchorConfig        map[string]processing.AnchorConfig
	anchorsFPN          map[string]*tensor.Dense
	numAnchors          map[string]int
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

	client.anchorConfig = map[string]processing.AnchorConfig{
		"32": {
			BaseSize:      16,
			Ratios:        ratio,
			Scales:        []float32{32, 16},
			AllowedBorder: 9999,
		},
		"16": {
			BaseSize:      16,
			Ratios:        ratio,
			Scales:        []float32{8, 4},
			AllowedBorder: 9999,
		},
		"8": {
			BaseSize:      16,
			Ratios:        ratio,
			Scales:        []float32{2, 1},
			AllowedBorder: 9999,
		},
	}

	client.fpnKeys = make([]string, 0)

	for s := range client.anchorConfig {
		client.fpnKeys = append(client.fpnKeys, fmt.Sprintf("stride%s", s))
	}

	client.anchorsFPN = make(map[string]*tensor.Dense)
	fpn, err := processing.GenerateAnchorsFPN2(false, client.anchorConfig)
	if err != nil {
		return nil, err
	}
	for idx, _ := range client.fpnKeys {
		client.anchorsFPN[client.fpnKeys[idx]] = fpn[idx]
	}

	client.numAnchors = make(map[string]int)
	anchorShape := make([]int, 0)

	for _, v := range client.anchorsFPN {
		anchorShape = append(anchorShape, v.Shape()[0])
	}

	for idx := range len(client.anchorsFPN) {
		client.numAnchors[client.fpnKeys[idx]] = anchorShape[idx]
	}

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

func (c *FaceDetectionClient) Infer(img gocv.Mat) error {
	preprocessedImg, preprocessedParam, err := c.preprocess(img)
	if err != nil {
		return err
	}
	fmt.Println("preprocessedParam", preprocessedParam)

	imgShape := preprocessedImg.Size()
	imgTensors := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(1, 3, imgShape[0], imgShape[1]),
	)

	for z := range 3 {
		for y := range imgShape[0] {
			for x := range imgShape[1] {
				err := imgTensors.SetAt((float32(preprocessedImg.GetVecbAt(y, x)[2-z])/c.pixelScale-c.pixelMeans[2-z])/c.pixelStds[2-z], 0, z, y, x)
				if err != nil {
					return err
				}
			}
		}
	}
	//fmt.Println(imgTensors)

	// Infer
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
		return err
	}
	netOut := make([]*tensor.Dense, len(c.ModelConfig.Config.Output))
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

		for subIdx, cfg := range c.ModelConfig.Config.Output {
			if out.Name == cfg.Name {
				netOut[subIdx] = outTensors
			}
		}
	}

	symIdx := 0
	for idx, s := range c.featStrideFPN {
		scores, err := netOut[symIdx].Slice(nil, tensor.S(c.numAnchors[fmt.Sprintf("stride%d", s)], netOut[symIdx].Shape()[2]-1), nil, nil)
		if err != nil {
			return err
		}
		bboxDeltas := netOut[symIdx+1]
		height, width := bboxDeltas.Shape()[2], bboxDeltas.Shape()[3]
		A := c.numAnchors[fmt.Sprintf("stride%d", s)]
		K := height * width
		anchorsFPN := c.anchorsFPN[fmt.Sprintf("stride%d", s)]
		anchors, err := rcnn.Anchors(height, width, s, anchorsFPN)
		if err != nil {
			return err
		}

		err = anchors.Reshape(K*A, 4)
		if err != nil {
			return err
		}

		err = scores.T(0, 2, 3, 1)
		if err != nil {
			return err
		}

		resizedScores := scores.Clone()
		err = resizedScores.(*tensor.Dense).Reshape(scores.DataSize(), 1)
		if err != nil {
			return err
		}

		err = bboxDeltas.T(0, 2, 3, 1)
		if err != nil {
			return err
		}

		fmt.Println(idx, bboxDeltas)

	}

	return nil
}
