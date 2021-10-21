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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/diag_kernel.h"

namespace oneflow {
namespace {

template<typename T>
struct DiagFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, T* out_buf, const T* in_buf, int32_t size, int32_t stride,
                  int32_t in_dim) {
    if (in_dim == 1) {
      FOR_RANGE(int32_t, i, 0, size) { out_buf[i * stride] = in_buf[i]; }
    } else {
      FOR_RANGE(int32_t, i, 0, size) { out_buf[i] = in_buf[i * stride]; }
    }
  }
};

template<typename T>
struct DiagGradFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, T* dx_buf, const T* dy_buf, int32_t dx_cnt, int32_t dy_cnt,
                  int32_t stride, int32_t in_dim) {
    if (in_dim == 1) {
      FOR_RANGE(int32_t, i, 0, dx_cnt) { dx_buf[i] = dy_buf[i * stride]; }
    } else {
      FOR_RANGE(int32_t, i, 0, dy_cnt) { dx_buf[i * stride] = dy_buf[i]; }
    }
  }
};

}  // namespace

REGISTER_DIAG_KERNELS(DeviceType::kCPU, float);
REGISTER_DIAG_KERNELS(DeviceType::kCPU, double);
REGISTER_DIAG_KERNELS(DeviceType::kCPU, int8_t);
REGISTER_DIAG_KERNELS(DeviceType::kCPU, int32_t);
REGISTER_DIAG_KERNELS(DeviceType::kCPU, int64_t);

}  // namespace oneflow
