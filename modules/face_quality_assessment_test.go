package modules

import (
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"gorgonia.org/tensor"
	"testing"
)

func TestNewFaceQualityAssessmentClient(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataSingleFace()
	assert.NoError(t, err)
	defer img.Close()
	detClient, err := NewFaceDetectionClient(tritonClient, config.DefaultRetinaFaceDetectionParams)
	assert.NoError(t, err)

	det, kpss, err := detClient.Infer(*img)
	assert.NoError(t, err)

	selectionClient := NewFaceSelectionClient(config.DefaultFaceSelectionParams)

	selectedFaceBox, selectedFacePoint, err := selectionClient.Infer(*img, det, kpss, utils.RefPointer(false))
	assert.NoError(t, err)

	faceBoxesS, err := selectedFaceBox.Slice(tensor.S(0, 4))
	assert.NoError(t, err)

	faceBoxes := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(selectedFaceBox.Shape()...))

	err = tensor.Copy(faceBoxes, faceBoxesS)
	assert.NoError(t, err)

	alignClient := NewFaceAlignmentClient(config.DefaultFaceAlignParams)
	alignedImg, err := alignClient.Infer(*img, selectedFaceBox, selectedFacePoint)
	assert.NoError(t, err)

	faceQualityAssessment, err := NewFaceQualityAssessmentClient(tritonClient, config.DefaultFaceQualityAssessmentParams)
	assert.NoError(t, err)

	_, _, err = faceQualityAssessment.Infer([]gocv.Mat{*alignedImg})
	assert.NoError(t, err)

}
