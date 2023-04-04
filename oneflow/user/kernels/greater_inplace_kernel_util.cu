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

template<typename T, typename ValueT>
__global__ void ScalarGreaterInplacForwardGpu(const int64_t n, const T* x, const Scalar operand,
                                              T* out) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    out[i] = x[i] > static_cast<T>(operand.Value<ValueT>()) ? static_cast<T>(1) : static_cast<T>(0);
  }
}

template<>
__global__ void ScalarGreaterInplacForwardGpu<half, int64_t>(const int64_t n, const half* x,
                                                             const Scalar operand, half* out) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    float operator_value = static_cast<float>(operand.Value<int64_t>());
    out[i] = x[i] > __float2half(operator_value) ? static_cast<half>(1) : static_cast<half>(0);
  }
}

template<>
__global__ void ScalarGreaterInplacForwardGpu<half, double>(const int64_t n, const half* x,
                                                            const Scalar operand, half* out) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    float operator_value = static_cast<float>(operand.Value<double>());
    out[i] = x[i] > __float2half(operator_value) ? static_cast<half>(1) : static_cast<half>(0);
  }
}

}  // namespace

template<typename T>
struct GreaterInplaceKernelUtil<DeviceType::kCUDA, T> {
  static void Forward(ep::Stream* stream, const int64_t n, const T* x, const T* y, T* out) {
    RUN_CUDA_KERNEL((GreaterInplacForwardGpu<T>), stream, n, n, x, y, out);
  }
};

template<typename T, typename ValueT>
struct ScalarGreaterInplaceKernelUtil<DeviceType::kCUDA, T, ValueT> {
  static void Forward(ep::Stream* stream, const int64_t n, const T* x, const Scalar operand,
                      T* out) {
    RUN_CUDA_KERNEL((ScalarGreaterInplacForwardGpu<T, ValueT>), stream, n, n, x, operand, out);
  }
};

#define INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CUDA(data_type, other) \
  template struct GreaterInplaceKernelUtil<DeviceType::kCUDA, data_type>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CUDA,
                     GREATER_INPLACE_DATA_TYPE_SEQ_CUDA)
#undef INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CUDA

#define INSTANTIATE_SCALAR_GREATER_INPLACE_KERNEL_UTIL_CUDA(data_type, value_data_type)          \
  template struct ScalarGreaterInplaceKernelUtil<DeviceType::kCUDA, OF_PP_PAIR_FIRST(data_type), \
                                                 OF_PP_PAIR_FIRST(value_data_type)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SCALAR_GREATER_INPLACE_KERNEL_UTIL_CUDA,
                                 GREATER_INPLACE_DATA_TYPE_SEQ_CUDA, SCALAR_VALUE_DATA_TYPE_SEQ)
#undef INSTANTIATE_SCALAR_GREATER_INPLACE_KERNEL_UTIL_CUDA

}  // namespace oneflow
