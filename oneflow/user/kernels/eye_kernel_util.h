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
#ifndef ONEFLOW_USER_KERNELS_EYE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_EYE_KERNEL_UTIL_H_
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {
namespace user_op {

#define EYE_DATA_TYPE_SEQ    \
  FLOATING_DATA_TYPE_SEQ     \
  INT_DATA_TYPE_SEQ          \
  UNSIGNED_INT_DATA_TYPE_SEQ \
  BOOL_DATA_TYPE_SEQ

template<DeviceType device_type, typename T>
struct EyeFunctor final {
  void operator()(ep::Stream* stream, const int64_t& cols, const int64_t& rows, T* out);
};

template<typename T>
OF_DEVICE_FUNC void SetOneInDiag(const int64_t cols, const int64_t rows, T* out) {
  const T one = static_cast<T>(1);
  XPU_1D_KERNEL_LOOP(i, rows) {
    const int64_t index = i * cols + i;
    out[index] = one;
  }
}

#define INSTANTIATE_EYE_FUNCTOR(device_type_v, dtype_pair) \
  template struct EyeFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_EYE_KERNEL_UTIL_H_
