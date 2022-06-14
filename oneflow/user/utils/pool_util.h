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
#ifndef ONEFLOW_USER_UTILS_POOL_UTIL_H_
#define ONEFLOW_USER_UTILS_POOL_UTIL_H_
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"

namespace oneflow {

typedef small_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;
typedef small_vector<int32_t, SHAPE_MAX_AXIS_SIZE> FixedVector;

class Params3D {
 public:
  Params3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
           const std::string& padding, const std::vector<int32_t>& padding_before,
           const std::vector<int32_t>& padding_after, const std::vector<int32_t>& pool_size,
           const std::vector<int32_t>& strides, const bool ceil_mode);
  ~Params3D() = default;
  void Reset(const ShapeView& x_shape);

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
  std::string padding_;
  bool ceil_mode_;
  int64_t batch_num_;
  int64_t channel_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_UTILS_POOL_UTIL_H_
