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
#include "oneflow/user/kernels/where_kernel_util.h"

namespace oneflow {

template<typename T, typename CondT>
struct WhereKernelUtil<DeviceType::kCPU, T, CondT> {
  static void Where(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                    const T* rhs, T* out) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = static_cast<bool>(cond[i]) ? lhs[i] : rhs[i]; }
  }
  static void WhereXScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                           const T x_scalar, const T* rhs, T* out) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = static_cast<bool>(cond[i]) ? x_scalar : rhs[i]; }
  }
  static void WhereYScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                           const T* lhs, const T y_scalar, T* out) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = static_cast<bool>(cond[i]) ? lhs[i] : y_scalar; }
  }
  static void WhereXYScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                            const T x_scalar, const T y_scalar, T* out) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      out[i] = static_cast<bool>(cond[i]) ? x_scalar : y_scalar;
    }
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTOR, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)

}  // namespace oneflow
