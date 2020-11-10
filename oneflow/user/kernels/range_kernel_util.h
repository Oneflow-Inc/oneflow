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
#ifndef ONEFLOW_USER_KERNELS_RANGE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_RANGE_KERNEL_UTIL_H_
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

#define RANGE_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ    \
  FLOAT16_DATA_TYPE_SEQ     \
  INT_DATA_TYPE_SEQ

namespace user_op {
template<DeviceType device_type, typename T>
struct RangeFunctor final {
  void operator()(DeviceCtx* ctx, const int32_t start, const int32_t delta,
                  const int32_t range_elem_cnt, T* out);
};

template<typename T>
OF_DEVICE_FUNC void DoRange(const int32_t start, const int32_t delta, const int32_t range_elem_cnt,
                            T* out) {
  XPU_1D_KERNEL_LOOP(i, range_elem_cnt) { out[i] = start + i * delta; }
}

#define INSTANTIATE_RANGE_FUNCTOR(device_type_v, dtype_pair) \
  template struct RangeFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_RANGE_KERNEL_UTIL_H_
