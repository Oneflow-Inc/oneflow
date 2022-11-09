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
#ifndef ONEFLOW_USER_KERNELS_CTC_LOSS_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_CTC_LOSS_KERNEL_UTIL_H_

#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename TARGET, typename IDX>
struct CtcLossKernelUtil final {
  static void CtcLossForward(ep::Stream* stream, const T* log_probs_ptr, const TARGET* targets_ptr,
                             const IDX* input_lengths_ptr, const IDX* target_lengths_ptr,
                             T* alpha_ptr, T* loss_ptr,
                             NdIndexOffsetHelper<int64_t, 3>& input_helper,
                             NdIndexOffsetHelper<int64_t, 3>& alpha_helper,
                             const int64_t batch_size, const int64_t max_input_length,
                             const int64_t max_target_length, const int64_t blank,
                             const int32_t targets_ndim);

  static void CtcLossBackward(ep::Stream* stream, const T* grad_out_ptr, const T* loss_ptr,
                              const T* alpha_ptr, const T* log_probs_ptr, const TARGET* targets_ptr,
                              const IDX* input_lengths_ptr, const IDX* target_lengths_ptr,
                              T* beta_ptr, T* grad_ptr,
                              NdIndexOffsetHelper<int64_t, 3>& input_helper,
                              NdIndexOffsetHelper<int64_t, 3>& beta_helper,
                              const int64_t batch_size, const int64_t max_input_length,
                              const int64_t max_target_length, const int64_t num_labels,
                              const int64_t blank, const bool zero_infinity,
                              const int32_t targets_ndim);
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CTC_LOSS_KERNEL_UTIL_H_
