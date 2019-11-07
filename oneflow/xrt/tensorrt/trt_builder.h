#ifndef ONEFLOW_XRT_TENSORRT_TRT_BUILDER_H_
#define ONEFLOW_XRT_TENSORRT_TRT_BUILDER_H_

#include "NvInfer.h"
#include "glog/logging.h"

#include "oneflow/xrt/tensorrt/trt_shape.h"
#include "oneflow/xrt/tensorrt/trt_unique_ptr.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class enum TrtValueKind : int {
  kUndef = 0,
  kTensor = 1,
  kWeight = 2,
};

inline bool IsUndef(const TrtValueKind &kind) {
  return kind == TrtValueKind::kUndef;
}

inline bool IsTensor(const TrtValueKind &kind) {
  return kind == TrtValueKind::kTensor;
}

inline bool IsWeight(const TrtValueKind &kind) {
  return kind == TrtValueKind::kWeight;
}

class TrtBuilder {
 public:
  explicit TrtBuilder(const std::string &name);

  const std::string &name() const { return builder_name_; }

  bool AddParameter(const Parameter &param);
  bool AddParameter(int64_t handle, const Parameter &param);

  nvinfer1::ITensor *GetTensor(int64_t handle);

  nvinfer1::Weights GetWeight(int64_t handle);

  const TrtValueKind &ValueKind(int64_t handle) const {
    CHECK_GT(value_kinds_.count(handle), 0)
        << "Handle " << handle << " has not been built for this builder.";
    return value_kinds_.at(handle);
  }

  void CheckHasParameter(int64_t handle) const {
    CHECK_GT(params_.count(handle), 0)
        << "Parameter is not found for handle " << handle;
  }

  nvinfer1::IBuilder *builder() const { return builder_.get(); }

  nvinfer1::INetworkDefinition *network() const { return network_.get(); }

  // Returns handle for the added tensor.
  int64_t AddTensor(nvinfer1::ITensor *tensor);

  // Returns handle for the added weight.
  int64_t AddWeight(nvinfer1::Weights &weight);

 private:
  int64_t IncreaseHandle() { return ++next_handle_; }

 private:
  std::string builder_name_;

  // The next new handle number.
  int64_t next_handle_ = 0;

  nv::unique_ptr<nvinfer1::IBuilder> builder_;
  nv::unique_ptr<nvinfer1::INetworkDefinition> network_;

  util::Map<int64_t, TrtValueKind> value_kinds_;
  util::Map<int64_t, Parameter> params_;
  util::Map<int64_t, nvinfer1::ITensor *> tensors_;
  util::Map<int64_t, nvinfer1::Weights> weights_;
};

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_BUILDER_H_
