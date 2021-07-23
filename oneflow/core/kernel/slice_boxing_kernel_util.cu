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
#include "oneflow/core/kernel/slice_boxing_kernel_util.h"

namespace oneflow {

template<typename T>
__global__ void AddGpu(int64_t n, const T* a, const T* b, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = a[i] + b[i]; }
}

template<>
__global__ void AddGpu(int64_t n, const half* a, const half* b, half* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = __hadd(a[i], b[i]); }
}

template<typename T>
struct SliceBoxingKernelUtil<DeviceType::kGPU, T> {
  static void Add(DeviceCtx* ctx, int64_t n, const T* a, const T* b, T* out);
};

template<typename T>
void SliceBoxingKernelUtil<DeviceType::kGPU, T>::Add(DeviceCtx* ctx, int64_t n, const T* a,
                                                     const T* b, T* out) {
  AddGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, a, b, out);
}

template<>
void SliceBoxingKernelUtil<DeviceType::kGPU, float16>::Add(DeviceCtx* ctx, int64_t n,
                                                           const float16* a, const float16* b,
                                                           float16* out) {
  SliceBoxingKernelUtil<DeviceType::kGPU, half>::Add(ctx, n, reinterpret_cast<const half*>(a),
                                                     reinterpret_cast<const half*>(b),
                                                     reinterpret_cast<half*>(out));
}

#define INSTANTIATE_SLICE_BOXING_KERNEL_UTIL_GPU(type_cpp, type_proto) \
  template struct SliceBoxingKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SLICE_BOXING_KERNEL_UTIL_GPU,
                     ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
#undef INSTANTIATE_SLICE_BOXING_KERNEL_UTIL_GPU

}  // namespace oneflow
