package config

import (
	"gorgonia.org/tensor"
	"time"
)

const (
	FaceQualityClassBad = iota
	FaceQualityClassGood
	FaceQualityClassWearingMask
	FaceQualityClassWearingSunglasses
)

var QualityClassMapper = map[int]string{
	FaceQualityClassBad:               "Bad",
	FaceQualityClassGood:              "Good",
	FaceQualityClassWearingMask:       "WearingMask",
	FaceQualityClassWearingSunglasses: "WearingSunglasses",
}

type RetinaFaceDetectionParams struct {
	ModelName           string        `json:"model_name"`
	Timeout             time.Duration `json:"timeout"`
	ImageSize           [2]int        `json:"image_size"`
	MaxBatchSize        int           `json:"max_batch_size"`
	ConfidenceThreshold float32       `json:"confidence_threshold"`
	IOUThreshold        float32       `json:"iou_threshold"`
}

var DefaultRetinaFaceDetectionParams = &RetinaFaceDetectionParams{
	ModelName:           "face_detection_retina",
	Timeout:             20 * time.Second,
	ImageSize:           [2]int{640, 640},
	MaxBatchSize:        1,
	ConfidenceThreshold: 0.7,
	IOUThreshold:        0.45,
}

func NewRetinaFaceDetectionParams(modelName string, timeout time.Duration, imgSize [2]int, maxBatchSize int, confidenceThreshold, iouThreshold float32) *RetinaFaceDetectionParams {
	return &RetinaFaceDetectionParams{
		ModelName:           modelName,
		Timeout:             timeout,
		ImageSize:           imgSize,
		MaxBatchSize:        maxBatchSize,
		ConfidenceThreshold: confidenceThreshold,
		IOUThreshold:        iouThreshold,
	}
}

type FaceAlignParams struct {
	ImageSize         [2]int        `json:"image_size"`
	StandardLandmarks *tensor.Dense `json:"standard_landmarks"`
}

var DefaultFaceAlignParams = &FaceAlignParams{
	ImageSize: [2]int{112, 112},
	StandardLandmarks: tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			38.2946, 51.6963,
			73.5318, 51.5014,
			56.0252, 71.7366,
			41.5493, 92.3655,
			70.7299, 92.2041,
		}),
	),
}

func NewFaceAlignParams(imgSize [2]int, standardLandmarks *tensor.Dense) *FaceAlignParams {
	return &FaceAlignParams{
		ImageSize:         imgSize,
		StandardLandmarks: standardLandmarks,
	}
}

type ArcFaceRecognitionParams struct {
	ModelName string        `json:"model_name"`
	Timeout   time.Duration `json:"timeout"`
	ImageSize [2]int        `json:"image_size"`
	BatchSize int           `json:"batch_size"`
}

var DefaultArcFaceRecognitionParams = &ArcFaceRecognitionParams{
	ModelName: "face_identification",
	Timeout:   20 * time.Second,
	ImageSize: [2]int{112, 112},
	BatchSize: 1,
}

func NewArcFaceRecognitionParams(modelName string, timeout time.Duration, imgSize [2]int, batchSize int) *ArcFaceRecognitionParams {
	return &ArcFaceRecognitionParams{
		ModelName: modelName,
		Timeout:   timeout,
		ImageSize: imgSize,
		BatchSize: batchSize,
	}
}

type FaceQualityParams struct {
	ModelName string        `json:"model_name"`
	Timeout   time.Duration `json:"timeout"`
	ImageSize [2]int        `json:"image_size"`
	BatchSize int           `json:"batch_size"`
	Threshold float32       `json:"threshold"`
}

var DefaultFaceQualityParams = &FaceQualityParams{
	ModelName: "face_quality",
	Timeout:   20 * time.Second,
	ImageSize: [2]int{112, 112},
	BatchSize: 1,
	Threshold: 0.5,
}

func NewFaceQualityParams(modelName string, timeout time.Duration, imgSize [2]int, batchSize int, threshold float32) *FaceQualityParams {
	return &FaceQualityParams{
		ModelName: modelName,
		Timeout:   timeout,
		ImageSize: imgSize,
		BatchSize: batchSize,
		Threshold: threshold,
	}
}

type FaceSelectionParams struct {
	MarginCenterLeftRatio   float32 `json:"margin_center_left_ratio"`
	MarginCenterRightRatio  float32 `json:"margin_center_right_ratio"`
	MarginEdgeRatio         float32 `json:"margin_edge_ratio"`
	MinimumFaceRatio        float32 `json:"minimum_face_ratio"`
	MinimumWidthHeightRatio float32 `json:"minimum_width_height_ratio"`
	MaximumWidthHeightRatio float32 `json:"maximum_width_height_ratio"`
}

var DefaultFaceSelectionParams = &FaceSelectionParams{
	MarginCenterLeftRatio:   0.3,
	MarginCenterRightRatio:  0.3,
	MarginEdgeRatio:         0.1,
	MinimumFaceRatio:        0.0075,
	MinimumWidthHeightRatio: 0.65,
	MaximumWidthHeightRatio: 1.1,
}

func NewFaceSelectionParams(marginCenterLeftRatio, marginCenterRightRatio, marginEdgeRatio, minimumFaceRatio, minimumWidthHeightRatio, maximumWidthHeightRatio float32) *FaceSelectionParams {
	return &FaceSelectionParams{
		MarginCenterLeftRatio:   marginCenterLeftRatio,
		MarginCenterRightRatio:  marginCenterRightRatio,
		MarginEdgeRatio:         marginEdgeRatio,
		MinimumFaceRatio:        minimumFaceRatio,
		MinimumWidthHeightRatio: minimumWidthHeightRatio,
		MaximumWidthHeightRatio: maximumWidthHeightRatio,
	}
}