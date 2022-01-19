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
#include "oneflow/user/kernels/l1_l2_regularize_gradient_kernel_util.h"

namespace oneflow {

template<typename T>
struct L1L2RegularizeGradientKernelUtil<DeviceType::kCPU, T> {
  static void RegularizeGradient(ep::Stream* stream, int64_t n, const T* model, const T* model_diff,
                                 T* out, const T l1, const T l2) {
    FOR_RANGE(int64_t, i, 0, n) {
      const T model_val = model[i];
      out[i] = model_diff[i] + l1 * (model_val >= 0 ? 1 : -1) + l2 * model_val;
    }
  }
};

#define INSTANTIATE_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct L1L2RegularizeGradientKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_CPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_CPU

}  // namespace oneflow
