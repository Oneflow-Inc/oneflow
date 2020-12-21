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
#ifndef ONEFLOW_USER_KERNELS_UNFOLD_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_UNFOLD_KERNEL_UTIL_H_
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class UnfoldKernelUtil {
 public:
  static void CFirstForward(const DeviceCtx* device_ctx, const Shape& in, const Shape& out_5d,
                            const Shape& out, const std::vector<int32_t>& kernel_size,
                            const std::vector<int32_t>& strides,
                            const std::vector<int32_t>& dilation_rate,
                            const std::vector<int32_t>& padding_before, const T* data_im,
                            T* data_col);

  static void CFirstBackward(const DeviceCtx* device_ctx, const Shape& in, const Shape& out_5d,
                             const Shape& out, const std::vector<int32_t>& kernel_size,
                             const std::vector<int32_t>& strides,
                             const std::vector<int32_t>& dilation_rate,
                             const std::vector<int32_t>& padding_before, const T* data_col,
                             T* data_im);
};

#define INSTANTIATE_UNFOLD_KERNEL_UTIL(device, dtype) \
  template class UnfoldKernelUtil<device, dtype>;
}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_UNFOLD_KERNEL_UTIL_H_
