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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/model_update_kernel_util.h"
#include "oneflow/user/kernels/multi_tensor_model_update_kernel_util.h"

namespace oneflow {

template<typename T, typename G>
struct MultiTensorSGDUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     TensorTupleParams<T, G, 2> tensor_tuple_params);
};

template<typename T, typename G>
void MultiTensorSGDUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, TensorTupleParams<T, G, 2> tensor_tuple_params) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  for (int64_t tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
    const int64_t tensor_elem_cnt = tensor_tuple_params.sizes[tensor_idx];
    for (int64_t i = 0; i < tensor_elem_cnt; i++) {
      SGDUpdateFunctor<T, G>()(tensor_tuple_params.model_diff_addresses[tensor_idx] + i,
                               tensor_tuple_params.model_addresses[0][tensor_idx] + i, scale, l1,
                               l2, weight_decay, learning_rate_val);
    }
  }
}

template struct MultiTensorSGDUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct MultiTensorSGDUpdateKernelUtil<DeviceType::kCPU, double, double>;

}  // namespace oneflow
