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
#ifdef WITH_CUDA
#include "cuda_runtime.h"
#endif
#include "oneflow/xrt/tensorrt/trt_builder.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

nvinfer1::ITensor *TrtBuilder::GetTensor(int64_t handle) {
  const TrtValueKind &kind = ValueKind(handle);
  CHECK(IsUndefKind(kind) || IsTensorKind(kind))
      << "Value should be undefined or tensor for handle " << handle;
  nvinfer1::ITensor *tensor = nullptr;
  if (IsUndefKind(kind)) {
    CheckHasParameter(handle);
    const Parameter &param = params_.at(handle);
    const char *name = param.name().c_str();
    // Convert data type and shape.
    TrtShape shape(param.shape(), param.data_type());
    tensor = network_->addInput(name, shape.data_type(), shape.shape());
    tensors_[handle] = tensor;
    value_kinds_[handle] = TrtValueKind::kTensor;
  }
  return tensors_.at(handle);
}

nvinfer1::Weights &TrtBuilder::GetWeight(int64_t handle) {
  const TrtValueKind &kind = ValueKind(handle);
  CHECK(IsUndefKind(kind) || IsWeightKind(kind))
      << "Value should be undefined or weight for handle " << handle;
  if (IsUndefKind(kind)) {
    CheckHasParameter(handle);
    const Parameter &param = params_.at(handle);
    // Convert data type and shape.
    TrtShape shape(param.shape(), param.data_type());
    // Weights passed to TensorRT layers are in host memory.
    size_t num_bytes = shape.count() * SizeOf(param.data_type());
    auto *host_data = new std::vector<uint8_t>(num_bytes);
#ifdef WITH_CUDA
    CHECK_EQ(cudaSuccess,
             cudaMemcpy(host_data->data(), param.data(), num_bytes, cudaMemcpyDefault));
#else
    LOG(FATAL) << "Recompile the project with CUDA please.";
#endif
    CHECK_EQ(host_weights_.count(param.name()), 0);
    host_weights_[param.name()] = std::shared_ptr<std::vector<uint8_t>>(host_data);

    nvinfer1::Weights weight;
    weight.type = shape.data_type();
    weight.values = host_data->data();
    weight.count = shape.count();
    weights_[handle] = weight;
    value_kinds_[handle] = TrtValueKind::kWeight;
  }
  return weights_.at(handle);
}

int64_t TrtBuilder::AddParameter(const Parameter &param) {
  int64_t handle = IncreaseHandle();
  params_[handle] = param;
  value_kinds_[handle] = TrtValueKind::kUndef;
  return handle;
}

int64_t TrtBuilder::AddTensor(nvinfer1::ITensor *tensor) {
  int handle = IncreaseHandle();
  tensors_[handle] = tensor;
  value_kinds_[handle] = TrtValueKind::kTensor;
  return handle;
}

int64_t TrtBuilder::AddWeight(nvinfer1::Weights &weight) {
  int handle = IncreaseHandle();
  weights_[handle] = weight;
  value_kinds_[handle] = TrtValueKind::kWeight;
  return handle;
}

nv::unique_ptr<nvinfer1::ICudaEngine> TrtBuilder::BuildCudaEngine() {
  auto build_config = nv::unique_ptr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
  return nv::unique_ptr<nvinfer1::ICudaEngine>(
      builder_->buildEngineWithConfig(*network_, *build_config));
}

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
