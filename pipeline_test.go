package go_faceid_pipeline

import (
	"fmt"
	"github.com/okieraised/go-faceid-pipeline/config"
	"github.com/okieraised/go-faceid-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"io"
	"os"
	"testing"
)

const (
	tritonTestURL = "10.124.68.173:8603"
)

func genTestDataSingleFace() (*gocv.Mat, error) {
	f, err := os.Open("./test_data/single.jpg")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	res, err := utils.ImageToOpenCV(content)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func genTestDataMultipleFace() (*gocv.Mat, error) {
	f, err := os.Open("./test_data/multiple.jpg")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	res, err := utils.ImageToOpenCV(content)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func genTestDataNoFace() (*gocv.Mat, error) {
	f, err := os.Open("./test_data/noface.jpg")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	res, err := utils.ImageToOpenCV(content)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func TestNewGeneralExtractPipeline_Single(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataSingleFace()
	assert.NoError(t, err)
	defer img.Close()

	client, err := NewGeneralExtractPipeline(tritonClient)
	assert.NoError(t, err)

	resp, err := client.ExtractFaceFeatures(*img, false)
	assert.NoError(t, err)
	assert.Equal(t, config.FaceQualityClassGood, resp.FaceQuality)

	fmt.Println("resp", resp)
}

func TestNewGeneralExtractPipeline_Multiple(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataMultipleFace()
	assert.NoError(t, err)
	defer img.Close()

	client, err := NewGeneralExtractPipeline(tritonClient)
	assert.NoError(t, err)

	resp, err := client.ExtractFaceFeatures(*img, false)
	assert.NoError(t, err)
	assert.Equal(t, config.FaceQualityClassGood, resp.FaceQuality)
	assert.Equal(t, 10, resp.FaceCount)

	fmt.Println("resp", resp)

}

func TestNewGeneralExtractPipeline_NoFace(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataNoFace()
	assert.NoError(t, err)
	defer img.Close()

	client, err := NewGeneralExtractPipeline(tritonClient)
	assert.NoError(t, err)

	resp, err := client.ExtractFaceFeatures(*img, false)
	assert.NoError(t, err)
	assert.Equal(t, config.FaceQualityClassBad, resp.FaceQuality)
	assert.Equal(t, 0, resp.FaceCount)
}

func TestNewAntiSpoofingExtractPipeline_Single(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataSingleFace()
	assert.NoError(t, err)
	defer img.Close()

	client, err := NewAntiSpoofingExtractPipeline(tritonClient)
	assert.NoError(t, err)

	resp, err := client.ExtractFaceFeatures(*img, false, false)
	assert.NoError(t, err)
	assert.Equal(t, config.FaceQualityClassGood, resp.FaceQuality)
	assert.Equal(t, 1, resp.FaceCount)

	fmt.Println("resp", resp)
}
