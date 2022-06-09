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
#ifndef ONEFLOW_USER_KERNELS_MULTI_TENSOR_MODEL_UPDATE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_MULTI_TENSOR_MODEL_UPDATE_KERNEL_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

// Kernel arg size has 4K limit.
constexpr int max_tensors[5] = {160, 80, 40, 20, 15};

template<typename T, typename G, int n>
struct TensorTupleParams {
  G* model_diff_addresses[max_tensors[n - 1]];
  T* model_addresses[n - 1][max_tensors[n - 1]];
  int64_t sizes[max_tensors[n - 1]];
  int32_t block_offset[max_tensors[n - 1]];  // use int32
};

template<DeviceType device_type, typename T, typename G>
struct MultiTensorSGDUpdateKernelUtil {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     TensorTupleParams<T, G, 2> tensor_tuple_params);
};

}  // namespace oneflow

#endif
