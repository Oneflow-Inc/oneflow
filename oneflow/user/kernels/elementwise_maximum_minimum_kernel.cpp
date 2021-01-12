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
#include "oneflow/user/kernels/elementwise_maximum_minimum_kernel_util.h"

namespace oneflow {
namespace user_op {

template<template<typename> class XumUtil, typename T>
struct RunKernelUtil<DeviceType::kCPU, XumUtil, T> final {
  static void BackwardKernel(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y,
                             T* dx, T* dy) {
    XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
      XumUtil<T>::Backward(&dz[idx], &x[idx], &y[idx], dx ? &dx[idx] : nullptr,
                           dy ? &dy[idx] : nullptr);
    }
  }

  static void ForwardKernel(DeviceCtx* ctx, int64_t elem_cnt, T* z, const T* x, const T* y) {
    FOR_RANGE(int64_t, idx, 0, elem_cnt) { z[idx] = XumUtil<T>()(x[idx], y[idx]); }
  }
};

#define REGISTER_XMUM_CPU_KERNELS(op_type_name, util, dtype)            \
  REGISTER_FORWARD_KERNEL(DeviceType::kCPU, op_type_name, util, dtype); \
  REGISTER_BACKWARD_KERNEL(DeviceType::kCPU, op_type_name, util, dtype);

REGISTER_XMUM_CPU_KERNELS("elementwise_maximum", MaximumUtil, float);
REGISTER_XMUM_CPU_KERNELS("elementwise_maximum", MaximumUtil, double);
REGISTER_XMUM_CPU_KERNELS("elementwise_minimum", MinimumUtil, float);
REGISTER_XMUM_CPU_KERNELS("elementwise_minimum", MinimumUtil, double);
}  // namespace user_op
}  // namespace oneflow
