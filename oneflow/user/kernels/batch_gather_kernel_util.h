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
#ifndef ONEFLOW_USER_KERNELS_BATCH_GATHER_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_BATCH_GATHER_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct BatchGatherKernelUtilImpl final {
  static void Forward(ep::Stream* stream, const T* in, const K* indices,
                      const Shape& flat_out_shape, int64_t gather_dim_size, T* out);
  static void Backward(ep::Stream* stream, const T* out_diff, const K* indices,
                       const Shape& flat_out_diff_shape, int64_t gather_dim_size, T* in_diff);
};

template<DeviceType device_type, typename T>
struct BatchGatherKernelUtil final {
  static void Forward(ep::Stream* stream, const Blob* in, const Blob* indices, Blob* out);
  static void Backward(ep::Stream* stream, const Blob* out_diff, const Blob* indices,
                       Blob* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_BATCH_GATHER_KERNEL_UTIL_H_
