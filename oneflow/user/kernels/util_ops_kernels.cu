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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/user/kernels/util_ops_kernels.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"

namespace oneflow {
namespace user_op {
#ifdef WITH_CUDA
template<typename T>
struct IsNanFunctor<DeviceType::kCUDA, T, std::enable_if_t<std::is_floating_point<T>::value>> {
  __device__ bool operator()(const T x) const { return isnan(x); }
};

template<typename T>
struct IsNanFunctor<DeviceType::kCUDA, T, std::enable_if_t<!std::is_floating_point<T>::value>> {
  __device__ bool operator()(const T x) const { return false; }
};

template<>
struct IsNanFunctor<DeviceType::kCUDA, half> {
  __device__ bool operator()(const half x) const {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    return __hisnan(x);
#else
    return isnan(__half2float(x));
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
  }
};

template<typename T>
struct IsInfFunctor<DeviceType::kCUDA, T, std::enable_if_t<std::is_floating_point<T>::value>> {
  __device__ bool operator()(const T x) const { return isinf(x); }
};

template<typename T>
struct IsInfFunctor<DeviceType::kCUDA, T, std::enable_if_t<!std::is_floating_point<T>::value>> {
  __device__ bool operator()(const T x) const { return false; }
};

template<>
struct IsInfFunctor<DeviceType::kCUDA, half> {
  __device__ bool operator()(const half x) const {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    return __hisinf(x);
#else
    return isinf(__half2float(x));
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
  }
};

#define REGISTER_UTIL_OPS_CUDA_KERNEL(device, dtype_pair)     \
  REGISTER_ISNAN_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair)) \
  REGISTER_ISINF_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair))

REGISTER_UTIL_OPS_CUDA_KERNEL(DeviceType::kCUDA, (half))
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UTIL_OPS_CUDA_KERNEL, (DeviceType::kCUDA),
                                 UTIL_OPS_DATA_TYPE_SEQ);
#endif  // WITH_CUDA
}  // namespace user_op
}  // namespace oneflow
