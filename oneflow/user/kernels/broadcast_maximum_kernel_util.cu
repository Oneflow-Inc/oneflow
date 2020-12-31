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
#ifdef WITH_CUDA
#include "oneflow/user/kernels/broadcast_maximum_kernel_util.h"
namespace oneflow {
namespace user_op {

template<typename T>
__global__ void MaximumBackwardGpuKernel(int64_t elem_cnt, const T* dz, const T* x, const T* y,
                                         T* dx, T* dy) {
  DoUpdateMaximumGrad<T>(elem_cnt, dz, x, y, dx, dy);
}

template<typename T>
struct MaximumBackwardFunctor<DeviceType::kGPU, T> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                  T* dy) {
    RUN_CUDA_KERNEL((MaximumBackwardGpuKernel<T>), ctx, elem_cnt, elem_cnt, dz, x, y, dx, dy);
  }
};

template struct MaximumBackwardFunctor<DeviceType::kGPU, float>;
template struct MaximumBackwardFunctor<DeviceType::kGPU, double>;

}  // namespace user_op
}  // namespace oneflow
#endif  // END WITH_CUDA
