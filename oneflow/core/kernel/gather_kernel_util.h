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
#ifndef ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct GatherKernelUtil final {
  static void Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out);
  static void Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out,
                      int64_t offset);
};

template<DeviceType device_type, typename T, typename K>
struct GatherKernelUtilImpl final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, int64_t offset);
};

#define GATHER_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_
