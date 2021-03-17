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
#ifndef ONEFLOW_USER_KERNELS_CUMSUM_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_CUMSUM_KERNEL_UTIL_H_

#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

#define CUMSUM_DATA_TYPE_CPU_SEQ \
  FLOATING_DATA_TYPE_SEQ         \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define CUMSUM_DATA_TYPE_GPU_SEQ \
  CUMSUM_DATA_TYPE_CPU_SEQ       \
  FLOAT16_DATA_TYPE_SEQ

namespace user_op {

template<DeviceType device_type, typename IN_T>
struct CumsumFunctor final {
  void operator()(DeviceCtx* ctx, int32_t instance_num, int32_t instance_size, int32_t post,
                  const bool exclusive, const bool reverse, const IN_T* input, IN_T* output);
};

template<typename IN_T>
OF_DEVICE_FUNC void DoCumsum(int32_t instance_num, int32_t instance_size, int32_t post,
                             const bool exclusive, const bool reverse, const IN_T* input,
                             IN_T* output) {
  XPU_1D_KERNEL_LOOP(i, instance_num) {
    const int32_t start_idx = reverse ? i % post + (i / post + 1) * instance_size * post - post
                                      : i % post + (i / post) * instance_size * post;
    output[start_idx] = 0;
    IN_T temp = 0;
    FOR_RANGE(int64_t, j, exclusive, instance_size) {
      int32_t out_index = reverse ? start_idx - j * post : start_idx + j * post;
      int32_t in_index =
          reverse ? start_idx - (j - exclusive) * post : start_idx + (j - exclusive) * post;
      temp += input[in_index];
      output[out_index] = temp;
    }
  }
}

// macros for functors instantiate(used by cumsum_kernel_util.cu and cumsum_kernel_uti.cpp)
#define INSTANTIATE_CUMSUM_FUNCTOR(device_type_v, dtype_pair) \
  template struct CumsumFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUMSUM_KERNEL_UTIL_H_