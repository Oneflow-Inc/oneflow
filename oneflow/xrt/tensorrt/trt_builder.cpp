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

    nvinfer1::Weights weight;
    weight.values = param.data();
    weight.type = shape.data_type();
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

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
