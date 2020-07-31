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
#ifndef ONEFLOW_XRT_TENSORRT_TRT_VALUE_H_
#define ONEFLOW_XRT_TENSORRT_TRT_VALUE_H_

#include "NvInfer.h"

#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/tensorrt/trt_builder.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

// TensorRT ITensor or Weights.
class TrtValue {
 public:
  TrtValue() = default;

  int handle() const { return handle_; }

  TrtValueKind ValueKind(TrtBuilder *builder) const {
    CHECK_EQ(builder_, builder);
    return builder_->ValueKind(handle_);
  }

  nvinfer1::ITensor *AsTensor(TrtBuilder *builder) {
    CHECK_EQ(builder_, builder);
    return builder_->GetTensor(handle_);
  }

  nvinfer1::Weights &AsWeight(TrtBuilder *builder) {
    CHECK_EQ(builder_, builder);
    return builder_->GetWeight(handle_);
  }

  inline static TrtValue Parameter(TrtBuilder *builder, const xrt::Parameter &param);

  inline static TrtValue Tensor(TrtBuilder *builder, nvinfer1::ITensor *tensor);

  inline static TrtValue Weight(TrtBuilder *builder, nvinfer1::Weights &weight);

 private:
  // Unique id for the `TrtValue`.
  int64_t handle_ = -1;
  TrtBuilder *builder_ = nullptr;
};

TrtValue TrtValue::Parameter(TrtBuilder *builder, const xrt::Parameter &param) {
  TrtValue trt_value;
  trt_value.handle_ = builder->AddParameter(param);
  trt_value.builder_ = builder;
  return std::move(trt_value);
}

TrtValue TrtValue::Tensor(TrtBuilder *builder, nvinfer1::ITensor *tensor) {
  TrtValue trt_value;
  trt_value.handle_ = builder->AddTensor(tensor);
  trt_value.builder_ = builder;
  return std::move(trt_value);
}

TrtValue TrtValue::Weight(TrtBuilder *builder, nvinfer1::Weights &weight) {
  TrtValue trt_value;
  trt_value.handle_ = builder->AddWeight(weight);
  trt_value.builder_ = builder;
  return std::move(trt_value);
}

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_VALUE_H_
