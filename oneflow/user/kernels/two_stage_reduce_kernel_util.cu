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
#include "oneflow/user/kernels/two_stage_reduce_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void DivideGpu(const int64_t n, const T* x, const K* count, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] / count[i]; }
}

template<typename T, typename K>
__global__ void MaskGpu(const int64_t n, const T* x, const K* mask, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = static_cast<T>(mask[i]) * x[i]; }
}

template<typename T, typename K>
__global__ void ScaleGpu(const int64_t n, const T* x, const K* scale, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * static_cast<T>(scale[i]); }
}

}  // namespace

template<typename T, typename K>
struct TwoStageReduceKernelUtil<DeviceType::kCUDA, T, K> {
  static void Divide(ep::Stream* stream, const int64_t n, const T* x, const K* count, T* y) {
    DivideGpu<T, K><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                      stream->As<ep::CudaStream>()->cuda_stream()>>>(n, x, count, y);
  }

  static void Mask(ep::Stream* stream, const int64_t n, const T* x, const K* mask, T* y) {
    MaskGpu<T, K><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                    stream->As<ep::CudaStream>()->cuda_stream()>>>(n, x, mask, y);
  }

  static void Scale(ep::Stream* stream, const int64_t n, const T* x, const K* scale, T* y) {
    ScaleGpu<T, K><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                     stream->As<ep::CudaStream>()->cuda_stream()>>>(n, x, scale, y);
  }
};

#define INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CUDA(data_type_pair, index_type_pair)          \
  template struct TwoStageReduceKernelUtil<DeviceType::kCUDA, OF_PP_PAIR_FIRST(data_type_pair), \
                                           OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CUDA,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ);
#undef INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CUDA

}  // namespace oneflow
