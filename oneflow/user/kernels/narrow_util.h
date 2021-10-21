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
#ifndef ONEFLOW_USER_KERNELS_NARROW_UTIL_H_
#define ONEFLOW_USER_KERNELS_NARROW_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct NarrowKernelUtil {
  static void Forward(DeviceCtx* ctx, const T* in, const Shape& flat_in_shape, T* out,
                      const int64_t& start, const int64_t& length);
  static void Backward(DeviceCtx* ctx, const T* dy, const Shape& flat_in_shape, T* dx,
                       const int64_t& start, const int64_t& length);
};

#define INSTANTIATE_NARROW_KERNEL_UTIL(device, dtype) \
  template struct NarrowKernelUtil<device, dtype>;

#define INSTANTIATE_NARROW_KERNEL_UTIL_WITH_DEVICE(device) \
  INSTANTIATE_NARROW_KERNEL_UTIL(device, float)            \
  INSTANTIATE_NARROW_KERNEL_UTIL(device, double)           \
  INSTANTIATE_NARROW_KERNEL_UTIL(device, int32_t)          \
  INSTANTIATE_NARROW_KERNEL_UTIL(device, int64_t)          \
  INSTANTIATE_NARROW_KERNEL_UTIL(device, int8_t)           \
  INSTANTIATE_NARROW_KERNEL_UTIL(device, uint8_t)

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_NARROW_UTIL_H_
