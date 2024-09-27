package go_faceid_pipeline

import (
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/modules"
	"github.com/okieraised/go-faceid-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
)

type GeneralExtractionResult struct {
	FacialFeatures  *tensor.Dense           `json:"facial_features"`
	FaceCount       int                     `json:"face_count"`
	FaceQuality     config.FaceQualityClass `json:"face_quality"`
	QualityScore    float32                 `json:"quality_score"`
	SelectedFaceBox *tensor.Dense           `json:"selected_face_box"`
}

type AntiSpoofingExtractionResult struct {
	FacialFeatures  *tensor.Dense           `json:"facial_features"`
	FaceCount       int                     `json:"face_count"`
	FaceQuality     config.FaceQualityClass `json:"face_quality"`
	QualityScore    float32                 `json:"quality_score"`
	SelectedFaceBox *tensor.Dense           `json:"selected_face_box"`
}

type GeneralExtractPipeline struct {
	tritonClient   *gotritonclient.TritonGRPCClient
	faceDetection  *modules.FaceDetectionClient
	faceSelection  *modules.FaceSelectionClient
	faceAlignment  *modules.FaceAlignmentClient
	faceQuality    *modules.FaceQualityClient
	faceExtraction *modules.FaceExtractionClient
}

// NewGeneralExtractPipeline initializes new faceid pipeline
func NewGeneralExtractPipeline(tritonClient *gotritonclient.TritonGRPCClient) (*GeneralExtractPipeline, error) {
	client := &GeneralExtractPipeline{}

	faceDetection, err := modules.NewFaceDetectionClient(tritonClient, config.DefaultRetinaFaceDetectionParams)
	if err != nil {
		return client, err
	}
	client.faceDetection = faceDetection

	faceSelection := modules.NewFaceSelectionClient(config.DefaultFaceSelectionParams)
	client.faceSelection = faceSelection

	faceAlignment := modules.NewFaceAlignmentClient(config.DefaultFaceAlignParams)
	client.faceAlignment = faceAlignment
	faceQuality, err := modules.NewFaceQualityClient(tritonClient, config.DefaultFaceQualityParams)
	if err != nil {
		return client, err
	}
	client.faceQuality = faceQuality

	faceExtraction, err := modules.NewFaceExtractionClient(tritonClient, config.DefaultArcFaceRecognitionParams)
	if err != nil {
		return client, err
	}
	client.faceExtraction = faceExtraction

	return client, nil
}

func (c *GeneralExtractPipeline) ExtractFaceFeatures(img gocv.Mat, isEnroll bool) (*GeneralExtractionResult, error) {
	var err error
	resp := &GeneralExtractionResult{}

	detections, keyPoints, err := c.faceDetection.Infer(img)
	if err != nil {
		return resp, err
	}
	resp.FaceCount = detections.Shape()[0]
	if resp.FaceCount == 0 {
		return resp, nil
	}

	selectedFaceBox, selectedFacePoint, err := c.faceSelection.Infer(img, detections, keyPoints, utils.RefPointer(isEnroll))
	if err != nil {
		return resp, err
	}

	if selectedFaceBox != nil {
		resp.SelectedFaceBox = selectedFaceBox
		alignedFaceImages, err := c.faceAlignment.Infer(img, selectedFaceBox, selectedFacePoint)
		if err != nil {
			return resp, err
		}

		defer func(m *gocv.Mat) {
			cErr := m.Close()
			if cErr != nil && err == nil {
				err = cErr
			}
		}(alignedFaceImages)

		qualityScores, qualityClasses, err := c.faceQuality.Infer([]gocv.Mat{*alignedFaceImages})
		if err != nil {
			return resp, err
		}
		resp.QualityScore = qualityScores[0]
		resp.FaceQuality = config.FaceQualityClass(qualityClasses[0])

		facialFeatures, err := c.faceExtraction.Infer([]gocv.Mat{*alignedFaceImages})
		if err != nil {
			return resp, err
		}
		resp.FacialFeatures = facialFeatures[0]
	}

	return resp, nil
}

type AntiSpoofingExtractPipeline struct {
	tritonClient   *gotritonclient.TritonGRPCClient
	faceDetection  *modules.FaceDetectionClient
	faceSelection  *modules.FaceSelectionClient
	faceAlignment  *modules.FaceAlignmentClient
	faceQuality    *modules.FaceQualityClient
	faceExtraction *modules.FaceExtractionClient
}

func NewAntiSpoofingExtractPipeline(tritonClient *gotritonclient.TritonGRPCClient) (*AntiSpoofingExtractPipeline, error) {
	client := &AntiSpoofingExtractPipeline{}

	faceDetection, err := modules.NewFaceDetectionClient(tritonClient, config.DefaultRetinaFaceDetectionParams)
	if err != nil {
		return client, err
	}
	client.faceDetection = faceDetection

	faceSelection := modules.NewFaceSelectionClient(config.DefaultFaceSelectionParams)
	client.faceSelection = faceSelection

	faceAlignment := modules.NewFaceAlignmentClient(config.DefaultFaceAlignParams)
	client.faceAlignment = faceAlignment
	faceQuality, err := modules.NewFaceQualityClient(tritonClient, config.DefaultFaceQualityParams)
	if err != nil {
		return client, err
	}
	client.faceQuality = faceQuality

	faceExtraction, err := modules.NewFaceExtractionClient(tritonClient, config.DefaultArcFaceRecognitionParams)
	if err != nil {
		return client, err
	}
	client.faceExtraction = faceExtraction

	return client, nil
}

func (c *AntiSpoofingExtractPipeline) ExtractFaceFeatures(img gocv.Mat, isEnroll, spoofingControl bool) (*AntiSpoofingExtractionResult, error) {
	//var err error
	resp := &AntiSpoofingExtractionResult{}

	return resp, nil
}
