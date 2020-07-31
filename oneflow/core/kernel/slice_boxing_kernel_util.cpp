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
#include "oneflow/core/kernel/slice_boxing_kernel_util.h"

namespace oneflow {

template<typename T>
struct SliceBoxingKernelUtil<DeviceType::kCPU, T> {
  static void Add(DeviceCtx* ctx, int64_t n, const T* a, const T* b, T* out) {
    for (int64_t i = 0; i < n; ++i) { out[i] = a[i] + b[i]; }
  }
};

#define INSTANTIATE_SLICE_BOXING_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct SliceBoxingKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SLICE_BOXING_KERNEL_UTIL_CPU,
                     ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
#undef INSTANTIATE_SLICE_BOXING_KERNEL_UTIL_CPU

}  // namespace oneflow
