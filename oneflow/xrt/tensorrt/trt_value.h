#ifndef ONEFLOW_XRT_TENSORRT_TRT_VALUE_H_
#define ONEFLOW_XRT_TENSORRT_TRT_VALUE_H_

#include "NvInfer.h"

#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/tensorrt/trt_builder.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class TrtValue {
 public:
  static TrtValue Build(TrtBuilder *builder, int64_t handle,
                        const Parameter &param) {
    TrtValue trt_value;
    if (builder->AddParameter(handle, param)) {
      trt_value.handle_ = handle;
      trt_value.builder_ = builder;
      trt_value.kind_ = TrtValueKind::kUndef;
    }
    return std::move(trt_value);
  }

  TrtValue() : kind_(TrtValueKind::kUndef) {}

  nvinfer1::ITensor *AsTensor(TrtBuilder *builder) {
    CHECK_EQ(builder_, builder) << "Must take the same trt builder.";
    if (!IsUndef() && !IsTensor()) {
      LOG(FATAL) << "`AsTensor` is not allowed since the value has been "
                    "defined as Weight.";
    }
    kind_ = TrtValueKind::kTensor;
    return builder_->GetTensor(handle_);
  }

  nvinfer1::Weights *AsWeight(TrtBuilder *builder) {
    CHECK_EQ(builder_, builder) << "Must take the same trt builder.";
    if (!IsUndef() && !IsWeight()) {
      LOG(FATAL) << "`AsWeight` is not allowed since the value has been "
                    "defined as Tensor.";
    }
    kind_ = TrtValueKind::kWeight;
    return builder_->GetWeight(handle_);
  }

 private:
  TrtValueKind kind_;

  // Unique id for the `TrtValue`.
  int64_t handle_ = -1;
  TrtBuilder *builder_ = nullptr;
};

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_VALUE_H_
