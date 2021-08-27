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

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/diagonal_kernel.h"

namespace oneflow {
namespace {

template<typename T>
struct DiagonalFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, T* out_buf, const T* in_buf, int32_t size, int32_t dim1,
                  int32_t dim2) {
    FOR_RANGE(int32_t, index, 0, size * dim2) {
      int32_t i = index / dim2;
      int32_t j = index % dim2;
      out_buf[j * size + i] = in_buf[i * (dim1 + 1) * dim2 + j];
    }
  }
};

template<typename T>
struct DiagonalGradFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, T* dx_buf, const T* dy_buf, int32_t size, int32_t dim1,
                  int32_t dim2) {
    FOR_RANGE(int32_t, index, 0, size * dim2) {
      int32_t i = index / dim2;
      int32_t j = index % dim2;
      dx_buf[i * (dim1 + 1) * dim2 + j] = dy_buf[j * size + i];
    }
  }
};

}  // namespace

REGISTER_DIAGONAL_KERNELS(DeviceType::kCPU, float);
REGISTER_DIAGONAL_KERNELS(DeviceType::kCPU, double);
REGISTER_DIAGONAL_KERNELS(DeviceType::kCPU, int8_t);
REGISTER_DIAGONAL_KERNELS(DeviceType::kCPU, int32_t);
REGISTER_DIAGONAL_KERNELS(DeviceType::kCPU, int64_t);

}  // namespace oneflow