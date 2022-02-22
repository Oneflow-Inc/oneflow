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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/elementwise_maximum_minimum_kernel.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {
template<template<typename> class Opt, typename T>
__global__ void ElementwiseXimumGradGpuKernel(int64_t elem_cnt, const T* dz, const T* x, const T* y,
                                              T* dx, T* dy) {
  XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
    Opt<T>()(dz[idx], x[idx], y[idx], dx ? &dx[idx] : nullptr, dy ? &dy[idx] : nullptr);
  }
}

template<template<typename> class Opt, typename T>
struct ElemwiseXimumGradFunctor<DeviceType::kCUDA, Opt, T> final {
  void operator()(ep::Stream* stream, int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                  T* dy) {
    ElementwiseXimumGradGpuKernel<Opt, T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(elem_cnt, dz, x, y, dx, dy);
  }
};

template<template<typename> class Opt, typename T>
struct ElemwiseXimumFunctor<DeviceType::kCUDA, Opt, T> final {
  void operator()(ep::Stream* stream, int64_t elem_cnt, T* z, const T* x, const T* y) {
    OF_CUDA_CHECK(cuda::elementwise::Binary(Opt<T>(), elem_cnt, z, x, y,
                                            stream->As<ep::CudaStream>()->cuda_stream()));
  }
};
}  // namespace

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MAXIMUM_KERNELS, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MINIMUM_KERNELS, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ)
}  // namespace oneflow
#endif  // WITH_CUDA
