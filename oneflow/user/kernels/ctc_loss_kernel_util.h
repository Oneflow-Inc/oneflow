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

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename IDX>
struct CtcLossKernelUtil {
  static void CtcLossForward(DeviceCtx* ctx, const IDX batch_size, const T* log_probs_ptr,
                             const int* targets_ptr, const IDX* input_lengths_ptr,
                             const IDX* target_lengths_ptr, T* alpha_ptr, T* loss_ptr,
                             NdIndexOffsetHelper<IDX, 3> input_helper,
                             NdIndexOffsetHelper<IDX, 3> alpha_helper, IDX max_target_length,
                             const int blank, const bool zero_infinity);

  static void CtcLossBackward(DeviceCtx* ctx, const T* grad_out_ptr, const T* loss_ptr,
                              const T* alpha_ptr, const IDX batch_size, const T* log_probs_ptr,
                              const int* targets_ptr, const IDX* input_lengths_ptr,
                              const IDX* target_lengths_ptr, T* beta_ptr, T* grad_ptr,
                              NdIndexOffsetHelper<IDX, 3> input_helper,
                              NdIndexOffsetHelper<IDX, 3> beta_helper, IDX max_input_length,
                              IDX max_target_length, IDX num_labels, const int blank,
                              const bool zero_infinity);
};

#define INSTANTIATE_CTC_LOSS_FUNCTOR(device_type_v, log_probs_dtype_pair,                  \
                                     input_lengths_dtype_pair)                             \
  template struct CtcLossKernelUtil<device_type_v, OF_PP_PAIR_FIRST(log_probs_dtype_pair), \
                                    OF_PP_PAIR_FIRST(input_lengths_dtype_pair)>;
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CTC_LOSS_KERNEL_UTIL_H_
