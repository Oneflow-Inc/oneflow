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
#include "oneflow/core/ndarray/ndarray_apply_unary_core.h"
#include "oneflow/core/ndarray/unary_func.h"

namespace oneflow {

namespace {

template<typename T, template<typename> class unary_func>
__global__ void NdarrayApplyUnaryInplaceApplyGpu(T* ptr, size_t n) {
  NdarrayApplyUnaryCore<T, unary_func>::InplaceApply(ptr, n);
}

}  // namespace

template<typename T, template<typename> class unary_func>
struct NdarrayApplyUnaryCoreWrapper<DeviceType::kCUDA, T, unary_func> final {
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y) {
    size_t n = y.host_shape().HostElemNum();
    if (n == 0) { return; }
    RUN_CUDA_KERNEL((NdarrayApplyUnaryInplaceApplyGpu<T, unary_func>), stream, n, y.host_ptr(), n);
  }
};

}  // namespace oneflow
