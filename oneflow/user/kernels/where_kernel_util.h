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
#ifndef ONEFLOW_USER_KERNELS_WHERE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_WHERE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename CondT>
struct WhereKernelUtil {
  static void Where(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                    const T* rhs, T* out);
  static void WhereXScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                           const T x_scalar, const T* rhs, T* out);
  static void WhereYScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                           const T* lhs, const T y_scalar, T* out);
  static void WhereXYScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                            const T x_scalar, const T y_scalar, T* out);
};

#define INSTANTIATE_WHERE_FUNCTOR(device_type_v, dtype_pair, ctype_pair)       \
  template struct WhereKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                  OF_PP_PAIR_FIRST(ctype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_WHERE_KERNEL_UTIL_H_
