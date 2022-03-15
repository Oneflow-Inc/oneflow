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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/eye_kernel_util.h"

namespace oneflow {

namespace user_op {

template<typename T>
__global__ void EyeForwardGpuKernel(const int64_t cols, const int64_t rows, T* out) {
  SetOneInDiag(cols, rows, out);
}

template<typename T>
struct EyeFunctor<DeviceType::kCUDA, T> final {
  void operator()(ep::Stream* stream, const int64_t& cols, const int64_t& rows, T* out) {
    RUN_CUDA_KERNEL((EyeForwardGpuKernel<T>), stream, rows, cols, rows, out);
  }
};
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_EYE_FUNCTOR, (DeviceType::kCUDA), EYE_DATA_TYPE_SEQ);
}  // namespace user_op
}  // namespace oneflow

#endif  // End WITH_CUDA
