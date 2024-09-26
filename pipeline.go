package go_faceid_pipeline

import "gocv.io/x/gocv"

type ExtractPipeline struct {
}

func NewExtractPipeline() *ExtractPipeline {
	return &ExtractPipeline{}
}

func (c *ExtractPipeline) ExtractFaceFeatures(img gocv.Mat, isEnroll bool) {

}
