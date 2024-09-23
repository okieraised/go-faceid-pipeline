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
	"io"
	"os"
	"testing"
)

const (
	tritonTestURL = "127.0.0.1:8603"
)

func genTestData() (*gocv.Mat, error) {
	f, err := os.Open("../test_data/test.png")
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

func TestNewFaceDetectionClient(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	img, err := genTestData()
	assert.NoError(t, err)
	defer img.Close()

	client, err := NewFaceDetectionClient(tritonClient, config.DefaultRetinaFaceDetectionParams)
	assert.NoError(t, err)

	err = client.Infer(*img)
	assert.NoError(t, err)

}
