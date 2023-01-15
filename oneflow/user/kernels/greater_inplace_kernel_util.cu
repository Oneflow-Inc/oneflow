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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/user/kernels/greater_inplace_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GreaterInplacForwardGpu(const int64_t n, const T* x, const T* y, T* out) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    out[i] = x[i] > y[i] ? static_cast<T>(1) : static_cast<T>(0);
  }
}

template<typename T>
__global__ void ScalarGreaterInplacForwardGpu(const int64_t n, const T* x, const Scalar* operand,
                                              T* out) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    out[i] = x[i] > operand->As<T>() ? static_cast<T>(1) : static_cast<T>(0);
  }
}

template<>
__global__ void ScalarGreaterInplacForwardGpu<half>(const int64_t n, const half* x,
                                                    const Scalar* operand, half* out) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    out[i] =
        x[i] > __float2half(operand->As<float>()) ? static_cast<half>(1) : static_cast<half>(0);
  }
}

}  // namespace

template<typename T>
struct GreaterInplaceKernelUtil<DeviceType::kCUDA, T> {
  static void Forward(ep::Stream* stream, const int64_t n, const T* x, const T* y, T* out) {
    RUN_CUDA_KERNEL((GreaterInplacForwardGpu<T>), stream, n, n, x, y, out);
  }

  static void ScalarForward(ep::Stream* stream, const int64_t n, const T* x, const Scalar* operand,
                            T* out) {
    RUN_CUDA_KERNEL((ScalarGreaterInplacForwardGpu<T>), stream, n, n, x, operand, out);
  }
};

#define INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CUDA(cpp_data_type, data_type) \
  template struct GreaterInplaceKernelUtil<DeviceType::kCUDA, cpp_data_type>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CUDA,
                     FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ)

#undef INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CUDA

}  // namespace oneflow
