#ifndef ONEFLOW_XRT_OPENVINO_INFERENCE_ENGINE_DATA_DESC_H_
#define ONEFLOW_XRT_OPENVINO_INFERENCE_ENGINE_DATA_DESC_H_

#include "glog/logging.h"

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace xrt {
namespace openvino {

inline InferenceEngine::Precision DataTypeToPrecision(const DataType &data_type) {
  switch (data_type) {
    case oneflow::kFloat: return InferenceEngine::Precision::FP32;
    case oneflow::kFloat16: return InferenceEngine::Precision::FP16;
    case oneflow::kInt8: return InferenceEngine::Precision::I8;
    case oneflow::kInt32: return InferenceEngine::Precision::I32;
    case oneflow::kInt64: return InferenceEngine::Precision::I64;
    default: {
      LOG(FATAL) << "Unsupported Precision for Openvino.";
      return InferenceEngine::Precision::FP32;
    }
  }
}

inline InferenceEngine::Layout ShapeToLayout(const Shape &shape) {
  switch (shape.NumAxes()) {
    case 1: return InferenceEngine::Layout::C;
    case 2: return InferenceEngine::Layout::NC;
    case 3: return InferenceEngine::Layout::CHW;
    case 4: return InferenceEngine::Layout::NCHW;
    case 5: return InferenceEngine::Layout::NCDHW;
    default: {
      LOG(FATAL) << "Unsupported Layout for Openvino.";
      return InferenceEngine::Layout::NCHW;
    }
  }
}

class InferenceEngineDataDesc {
 public:
  InferenceEngineDataDesc() = default;

  InferenceEngineDataDesc(const Shape &shape, const DataType &data_type)
      : precision_(DataTypeToPrecision(data_type)), layout_(ShapeToLayout(shape)) {
    for (int i = 0; i < shape.NumAxes(); ++i) { dim_vec_.push_back(shape.At(i)); }
  }

  const InferenceEngine::Precision &precision() const { return precision_; }
  const InferenceEngine::Layout &layout() const { return layout_; }

  const std::vector<size_t> &dims() const { return dim_vec_; }

 private:
  InferenceEngine::Precision precision_;
  InferenceEngine::Layout layout_;
  std::vector<size_t> dim_vec_;
};

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_OPENVINO_INFERENCE_ENGINE_DATA_DESC_H_
