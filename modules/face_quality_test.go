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
	"testing"
)

func TestNewFaceQualityClient_Single(t *testing.T) {
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

	alignClient := NewFaceAlignmentClient(config.DefaultFaceAlignParams)
	alignedImg, err := alignClient.Infer(*img, selectedFaceBox, selectedFacePoint)
	assert.NoError(t, err)

	qualityClient, err := NewFaceQualityClient(tritonClient, config.DefaultFaceQualityParams)
	assert.NoError(t, err)
	_, _, err = qualityClient.Infer([]gocv.Mat{*alignedImg})
	assert.NoError(t, err)

	alignedImg.Close()
}

func TestNewFaceQualityClient_Multiple(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataMultipleFace()
	assert.NoError(t, err)
	defer img.Close()
	detClient, err := NewFaceDetectionClient(tritonClient, config.DefaultRetinaFaceDetectionParams)
	assert.NoError(t, err)

	det, kpss, err := detClient.Infer(*img)
	assert.NoError(t, err)

	selectionClient := NewFaceSelectionClient(config.DefaultFaceSelectionParams)

	selectedFaceBox, selectedFacePoint, err := selectionClient.Infer(*img, det, kpss, utils.RefPointer(false))
	assert.NoError(t, err)

	alignClient := NewFaceAlignmentClient(config.DefaultFaceAlignParams)
	alignedImg, err := alignClient.Infer(*img, selectedFaceBox, selectedFacePoint)
	assert.NoError(t, err)

	qualityClient, err := NewFaceQualityClient(tritonClient, config.DefaultFaceQualityParams)
	assert.NoError(t, err)
	_, _, err = qualityClient.Infer([]gocv.Mat{*alignedImg})
	assert.NoError(t, err)

	alignedImg.Close()

}