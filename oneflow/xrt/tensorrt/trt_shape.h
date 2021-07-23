/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_XRT_TENSORRT_TRT_SHAPE_H_
#define ONEFLOW_XRT_TENSORRT_TRT_SHAPE_H_

#include "NvInfer.h"
#include "glog/logging.h"

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

inline nvinfer1::DataType DataTypeToTrtDataType(const DataType& data_type) {
  switch (data_type) {
    case oneflow::kFloat: return nvinfer1::DataType::kFLOAT;
    case oneflow::kInt8: return nvinfer1::DataType::kINT8;
    case oneflow::kInt32: return nvinfer1::DataType::kINT32;
    case oneflow::kFloat16: return nvinfer1::DataType::kHALF;
    default: {
      LOG(FATAL) << "Unsupported data type " << data_type << " for TensorRT.";
      return nvinfer1::DataType::kFLOAT;
    }
  }
}

inline nvinfer1::Dims ShapeToXrtDims(const Shape& shape) {
  CHECK_LE(shape.NumAxes(), 8) << "The maximum dimensions is 8 supported by TensorRT.";
  nvinfer1::Dims dims;
  dims.nbDims = shape.NumAxes();
  for (int i = 0; i < dims.nbDims; ++i) { dims.d[i] = shape.At(i); }
  return std::move(dims);
}

class TrtShape {
 public:
  TrtShape() = default;

  explicit TrtShape(const Shape& shape, const DataType& data_type)
      : dims_(ShapeToXrtDims(shape)), data_type_(DataTypeToTrtDataType(data_type)) {}

  int64_t count() const {
    if (count_ < 0) {
      count_ = 1;
      for (int i = 0; i < dims_.nbDims; ++i) { count_ *= dims_.d[i]; }
    }
    return count_;
  }

  const nvinfer1::DataType& data_type() const { return data_type_; }

  const nvinfer1::Dims& shape() const { return dims_; }

 private:
  mutable int64_t count_ = -1;

  nvinfer1::Dims dims_;
  nvinfer1::DataType data_type_;
};

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_SHAPE_H_
