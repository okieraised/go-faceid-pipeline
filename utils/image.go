package utils

import (
	"errors"
	"fmt"
	"gocv.io/x/gocv"
)

// ImageToOpenCV converts the raw image into OpenCV Matrix
func ImageToOpenCV(bImage []byte) (*gocv.Mat, error) {
	dstMat := gocv.Mat{}
	srcMat, err := gocv.IMDecode(bImage, gocv.IMReadUnchanged)
	if err != nil {
		return &gocv.Mat{}, err
	}

	// Add the rows, columns, and number of channel to the dimension
	dimension := []int{}
	dimension = append(dimension, srcMat.Size()...)
	dimension = append(dimension, srcMat.Channels())

	if len(dimension) < 3 {
		return &dstMat, errors.New(fmt.Sprintf("invalid number of dimension: %d", len(dimension)))
	}

	if dimension[2] == 4 { // RGBA
		gocv.CvtColor(srcMat, &dstMat, gocv.ColorBGRAToBGR)
	} else if dimension[2] == 1 { // Grayscale
		gocv.CvtColor(srcMat, &dstMat, gocv.ColorGrayToBGR)
	} else {
		dstMat = srcMat
	}
	return &dstMat, nil
}
