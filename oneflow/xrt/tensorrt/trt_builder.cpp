#include "oneflow/xrt/tensorrt/trt_builder.h"
#include "oneflow/xrt/tensorrt/trt_logger.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

TrtBuilder::TrtBuilder(const std::string &name)
    : builder_name_(name), next_handle_(0) {
  nv::Logger logger(name);
  builder_.reset(nvinfer1::createInferBuilder(Logger));
  network_.reset(builder_->createNetwork());
}

bool TrtBuilder::AddParameter(const Parameter &param) {
  return AddParameter(next_handle_, param);
}

bool TrtBuilder::AddParameter(int64_t handle, const Parameter &param) {
  if (!value_kinds_.emplace(handle, TrtValueKind::kUndef).second) {
    LOG(FATAL) << "Handle " << handle
               << " is already exist in the builder. Please check it.";
  }
  params_[handle] = param;
  next_handle_ = std::max(handle + 1, next_handle_);
  return true /* add parameter successfully */;
}

nvinfer1::ITensor *TrtBuilder::GetTensor(int64_t handle) {
  const TrtValueKind &kind = ValueKind(handle);
  CHECK(IsUndef(kind) || IsTensor(kind))
      << "Value should be undefined or tensor for handle " << handle;
  nvinfer1::ITensor *tensor = nullptr;
  if (IsUndef(kind)) {
    CheckHasParameter(handle);
    const Parameter &param = params_.at(handle);
    // Convert data type and shape.
    TrtShape shape(param.shape(), param.data_type());
    tensor = network_->addInput(param.name(), shape().data_type(),
                                shape.shape());
    tensors_[handle] = tensor;
    value_kinds_[handle] = TrtValueKind::kTensor;
  }
  return tensors_.at(handle);
}

nvinfer1::Weights &TrtBuilder::GetWeight(int64_t handle) {
  const TrtValueKind &kind = ValueKind(handle);
  CHECK(IsUndef(kind) || IsWeight(kind))
      << "Value should be undefined or weight for handle " << handle;
  if (IsUndef(kind)) {
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

int64_t TrtBuilder::AddTensor(nvinfer1::ITensor *tensor) {
  int handle = next_handle_;
  IncreaseHandle();
  tensors_[handle] = tensor;
  value_kinds_[handle] = TrtValueKind::kTensor;
  return handle;
}

int64_t TrtBuilder::AddWeight(nvinfer1::Weights &weight) {
  int handle = next_handle_;
  IncreaseHandle();
  weights_[handle] = weight;
  value_kinds_[handle] = TrtValueKind::kWeight;
  return handle;
}

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_BUILDER_H_
