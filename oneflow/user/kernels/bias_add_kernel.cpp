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
#include "oneflow/user/kernels/bias_add_kernel.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<typename T, typename Index>
struct BiasAddCalculation<DeviceType::kCPU, T, Index> {
  static void Invoke(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                     const T* x, const T* bias, T* y) {
    const Shape in_out_shape({outer_size, bias_size, inner_size});
    const Shape bias_shape({1, bias_size, 1});
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastAdd(ctx, XpuVarNdarray<T>(in_out_shape, y),
                                                   XpuVarNdarray<const T>(in_out_shape, x),
                                                   XpuVarNdarray<const T>(bias_shape, bias));
  }
};

REGISTER_BIAS_ADD_USER_KERNEL(CPU, float)
REGISTER_BIAS_ADD_USER_KERNEL(CPU, double)
REGISTER_BIAS_ADD_USER_KERNEL(CPU, int8_t)
REGISTER_BIAS_ADD_USER_KERNEL(CPU, int32_t)
REGISTER_BIAS_ADD_USER_KERNEL(CPU, int64_t)

}  // namespace oneflow
