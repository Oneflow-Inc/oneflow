#ifndef ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;
typedef fixed_vector<int32_t, SHAPE_MAX_AXIS_SIZE> FixedVector;

class Params3D {
 public:
  Params3D(const int32_t dim, const Shape& x_shape, const std::string& data_format,
           const std::string& padding, const std::vector<int32_t>& pool_size,
           const std::vector<int32_t>& strides);
  ~Params3D() = default;

  Shape GetYShape() const;

 private:
  int32_t dim_;
  FixedDimVector x_3d_;
  FixedDimVector y_3d_;
  FixedVector pool_size_;
  FixedVector strides_;
  FixedVector padding_;
  std::vector<int32_t> padding_before_;
  std::vector<int32_t> padding_after_;
  std::string data_format_;
  int64_t batch_num_;
  int64_t channel_num_;
};

class GPUPoolOpKernelState final {
 public:
  GPUPoolOpKernelState();
  ~GPUPoolOpKernelState() = default;

 private:
  const cudnnTensorDescriptor_t& cudnn_x_tensor_desc() const;
  const cudnnTensorDescriptor_t& cudnn_y_tensor_desc() const;
  const cudnnPoolingDescriptor_t& cudnn_pooling_desc() const;

  cudnnPoolingMode_t pooling_mode_;
  std::unique_ptr<CudnnTensorDesc> x_desc_;
  std::unique_ptr<CudnnTensorDesc> y_desc_;
  std::unique_ptr<cudnnPoolingDescriptor_t> pooling_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
