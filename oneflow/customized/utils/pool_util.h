#ifndef ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;
typedef fixed_vector<int32_t, SHAPE_MAX_AXIS_SIZE> FixedVector;

class Params3D {
 public:
  Params3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
           const std::string& padding, const std::vector<int32_t>& pool_size,
           const std::vector<int32_t>& strides);
  ~Params3D() = default;

  Shape GetYShape() const;
  Shape GetXShape5D() const;
  Shape GetYShape5D() const;

  const std::vector<int32_t>& pool_size_3d() const { return pool_size_3d_; }
  const std::vector<int32_t>& strides_3d() const { return strides_3d_; }
  const std::vector<int32_t>& padding_before_3d() const { return padding_before_3d_; }
  const std::vector<int32_t>& padding_after_3d() const { return padding_after_3d_; }

 private:
  int32_t dim_;
  FixedDimVector x_3d_;
  FixedDimVector y_3d_;
  std::vector<int32_t> pool_size_3d_;
  std::vector<int32_t> strides_3d_;
  std::vector<int32_t> padding_before_3d_;
  std::vector<int32_t> padding_after_3d_;
  std::string data_format_;
  int64_t batch_num_;
  int64_t channel_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
