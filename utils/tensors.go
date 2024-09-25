package utils

import (
	"fmt"
	"gorgonia.org/tensor"
	"sort"
)

func VStack(tensors []*tensor.Dense) (*tensor.Dense, error) {
	var nonEmptyTensors []*tensor.Dense
	for _, t := range tensors {
		shape := t.Shape()
		if shape[0] > 0 {
			nonEmptyTensors = append(nonEmptyTensors, t)
		}
	}

	if len(nonEmptyTensors) == 0 {
		return tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(0, 1)), nil
	}

	result, err := nonEmptyTensors[0].Concat(0, nonEmptyTensors[1:]...)
	if err != nil {
		return nil, fmt.Errorf("error concatenating tensors: %v", err)
	}

	return result, nil
}

func HStack(tensors []*tensor.Dense) (*tensor.Dense, error) {
	var nonEmptyTensors []*tensor.Dense
	for _, t := range tensors {
		shape := t.Shape()
		if shape[0] > 0 {
			nonEmptyTensors = append(nonEmptyTensors, t)
		}
	}

	if len(nonEmptyTensors) == 0 {
		return tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(0, 1)), nil
	}

	result, err := nonEmptyTensors[0].Concat(1, nonEmptyTensors[1:]...)
	if err != nil {
		return nil, fmt.Errorf("error concatenating tensors: %v", err)
	}

	return result, nil
}

func ArgSortDescending(t *tensor.Dense) ([]int, error) {
	shape := t.Shape()
	if len(shape) != 1 {
		return nil, fmt.Errorf("expected a 1D tensor, got shape %v", shape)
	}

	data := t.Data().([]float32)

	indices := make([]int, len(data))
	for i := range indices {
		indices[i] = i
	}

	sort.Slice(indices, func(i, j int) bool {
		return data[indices[i]] > data[indices[j]]
	})

	return indices, nil
}

func SelectRows1D(t *tensor.Dense, indices []int) (*tensor.Dense, error) {
	shape := t.Shape()
	if len(shape) != 1 {
		return nil, fmt.Errorf("expected a 1D tensor, got shape %v", shape)
	}
	num_rows := shape[0]

	selected_data := make([]float32, 0, len(indices))

	for _, idx := range indices {
		if idx >= num_rows {
			return nil, fmt.Errorf("index %d is out of bounds", idx)
		}
		row, err := t.Slice(tensor.S(idx), nil, nil)
		if err != nil {
			return nil, err
		}

		row_data := row.Data().([]float32)
		selected_data = append(selected_data, row_data...)
	}

	selected_tensor := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(len(indices)), tensor.WithBacking(selected_data))

	return selected_tensor, nil
}

func SelectRows2D(t *tensor.Dense, indices []int) (*tensor.Dense, error) {
	shape := t.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected a 2D tensor")
	}
	numCols := shape[1]

	selectedData := make([]float32, 0, len(indices)*numCols)

	for _, idx := range indices {
		row, err := t.Slice(tensor.S(idx), nil)
		if err != nil {
			return nil, err
		}

		switch row.Data().(type) {
		case []float32:
			rowData := row.Data().([]float32)
			selectedData = append(selectedData, rowData...)
		case float32:
			rowData := row.Data().(float32)
			selectedData = append(selectedData, rowData)
		}

	}

	selectedTensor := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(len(indices), numCols),
		tensor.WithBacking(selectedData),
	)

	return selectedTensor, nil
}

func SelectRows3D(t *tensor.Dense, indices []int) (*tensor.Dense, error) {
	shape := t.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("expected a 3D tensor, got shape %v", shape)
	}
	num_rows, num_cols, num_depth := shape[0], shape[1], shape[2]

	selected_data := make([]float32, 0, len(indices)*num_cols*num_depth)

	for _, idx := range indices {
		if idx >= num_rows {
			return nil, fmt.Errorf("index %d is out of bounds", idx)
		}
		row, err := t.Slice(tensor.S(idx), nil, nil)
		if err != nil {
			return nil, err
		}

		row_data := row.Data().([]float32)
		selected_data = append(selected_data, row_data...)
	}

	selected_tensor := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(len(indices), num_cols, num_depth), tensor.WithBacking(selected_data))

	return selected_tensor, nil
}

func TensorByIndices(t *tensor.Dense, indices []int) (*tensor.Dense, error) {
	shape := t.Shape()

	if len(shape) != 1 {
		return nil, fmt.Errorf("input tensor should be 1D, got shape %v", shape)
	}

	resultData := make([]float32, len(indices))

	for i, idx := range indices {
		element, err := t.At(idx)
		if err != nil {
			return nil, err
		}
		resultData[i] = element.(float32)
	}
	result := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(len(indices)), tensor.WithBacking(resultData))

	return result, nil
}
