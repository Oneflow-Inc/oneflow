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
#include "oneflow/user/kernels/two_stage_reduce_kernel_util.h"
#include "oneflow/core/common/data_type_seq.h"

namespace oneflow {

template<typename T, typename K>
struct TwoStageReduceKernelUtil<DeviceType::kCPU, T, K> {
  static void Divide(ep::Stream* stream, const int64_t n, const T* x, const K* count, T* y) {
    FOR_RANGE(int64_t, i, 0, n) { y[i] = x[i] / count[i]; }
  }

  static void Mask(ep::Stream* stream, const int64_t n, const T* x, const K* mask, T* y) {
    FOR_RANGE(int64_t, i, 0, n) { y[i] = static_cast<T>(mask[i]) * x[i]; }
  }

  static void Scale(ep::Stream* stream, const int64_t n, const T* x, const K* scale, T* y) {
    FOR_RANGE(int64_t, i, 0, n) { y[i] = x[i] * static_cast<T>(scale[i]); }
  }
};

#define INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CPU(data_type_pair, index_type_pair)          \
  template struct TwoStageReduceKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                           OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ);
#undef INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CPU

}  // namespace oneflow
