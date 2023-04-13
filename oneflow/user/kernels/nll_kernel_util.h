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
#ifndef ONEFLOW_USER_KERNELS_NLL_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_NLL_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct NLLKernelUtil {
  static void Forward(ep::Stream* stream, const int32_t num_samples, const K num_classes,
                      const K class_start, const K ignore_index, const T* input, const K* target,
                      const T* weight, T* out, T* out_weight);

  static void Backward(ep::Stream* stream, const int32_t num_samples, const K num_classes,
                       const K class_start, const K ignore_index, const T* out_grad,
                       const K* target, const T* weight, T* in_grad);
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_NLL_KERNEL_UTIL_H_
