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
#ifndef ONEFLOW_USER_KERNELS_BROADCAST_MINIMUM_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_BROADCAST_MINIMUM_KERNEL_UTIL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T>
struct MinimumBackwardFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                  T* dy);
};

template<typename T>
OF_DEVICE_FUNC void DoUpdateMinimumGrad(int64_t elem_cnt, const T* dz, const T* x, const T* y,
                                        T* dx, T* dy) {
  XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
    if (x[idx] < y[idx]) {
      dx[idx] = dz[idx];
    } else {
      dy[idx] = dz[idx];
    }
  }
}

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_BROADCAST_MINIMUM_KERNEL_H_