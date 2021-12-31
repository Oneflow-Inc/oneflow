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
#include "oneflow/user/kernels/sqrt_square_sum_kernel_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T, bool ONE_BLOCK>
__global__ void SqrtSquareSumGpu(int64_t n, const T* x, T* y) {
  T t_sum = 0;
  CUDA_1D_KERNEL_LOOP(i, n) { t_sum += x[i] * x[i]; }
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T b_sum = BlockReduce(temp_storage).Sum(t_sum);
  if (threadIdx.x == 0) {
    if (ONE_BLOCK) {
      *y = b_sum;
    } else {
      cuda::atomic::Add(y, b_sum);
    }
  }
}

}  // namespace

template<typename T>
struct SqrtSquareSumKernelUtil<DeviceType::kCUDA, T> {
  static void SqrtSquareSum(ep::Stream* stream, int64_t n, const T* x, T* y) {
    const int32_t num_blocks = BlocksNum4ThreadsNum(n);
    CHECK_GE(num_blocks, 0);
    if (num_blocks == 0) {
      Memset<DeviceType::kCUDA>(stream, y, 0, sizeof(T));
    } else if (num_blocks == 1) {
      SqrtSquareSumGpu<T, true>
          <<<1, kCudaThreadsNumPerBlock, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(n, x, y);
    } else {
      Memset<DeviceType::kCUDA>(stream, y, 0, sizeof(T));
      SqrtSquareSumGpu<T, false>
          <<<num_blocks, kCudaThreadsNumPerBlock, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              n, x, y);
    }

    // todo(launch sqrt kernel)
    
  }
};

#define INSTANTIATE_SQRT_SQUARE_SUM_KERNEL_UTIL_CUDA(type_cpp, type_proto) \
  template struct SqrtSquareSumKernelUtil<DeviceType::kCUDA, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SQRT_SQUARE_SUM_KERNEL_UTIL_CUDA, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_SQRT_SQUARE_SUM_KERNEL_UTIL_CUDA

}  // namespace oneflow
