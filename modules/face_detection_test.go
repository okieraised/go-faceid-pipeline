package modules

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
	f, err := os.Open("../test_data/single.png")
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
	f, err := os.Open("../test_data/multiple.jpg")
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
	f, err := os.Open("../test_data/noface.jpg")
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

func TestNewFaceDetectionClient_Single(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataSingleFace()
	assert.NoError(t, err)
	defer img.Close()
	client, err := NewFaceDetectionClient(tritonClient, config.DefaultRetinaFaceDetectionParams)
	assert.NoError(t, err)

	det, kpss, err := client.Infer(*img)
	assert.NoError(t, err)

	fmt.Println("det", det)
	fmt.Println("kpss", kpss)
}

func TestNewFaceDetectionClient_Multiple(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataMultipleFace()
	assert.NoError(t, err)
	defer img.Close()
	client, err := NewFaceDetectionClient(tritonClient, config.DefaultRetinaFaceDetectionParams)
	assert.NoError(t, err)

	det, kpss, err := client.Infer(*img)
	assert.NoError(t, err)

	fmt.Println("det", det)
	fmt.Println("kpss", kpss)
}

func TestNewFaceDetectionClient_NoFace(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestDataNoFace()
	assert.NoError(t, err)
	defer img.Close()
	client, err := NewFaceDetectionClient(tritonClient, config.DefaultRetinaFaceDetectionParams)
	assert.NoError(t, err)

	det, kpss, err := client.Infer(*img)
	assert.NoError(t, err)
	fmt.Println(det.Shape())
	fmt.Println(kpss.Shape())
}
