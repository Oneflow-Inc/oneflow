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
#include "oneflow/core/kernel/square_sum_kernel_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T, bool ONE_BLOCK>
__global__ void SquareSumGpu(int64_t n, const T* x, T* y) {
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

constexpr int64_t kMultiSquareSumMaxSize = 64;

template<typename T>
struct MultiSquareSumParams {
  SquareSumParam<T> params[kMultiSquareSumMaxSize];
  int32_t size;
};

template<typename T>
__global__ void MultiSquareSumGpu(const MultiSquareSumParams<T> params, T* y) {
  T t_sum = 0;
  for (int i = 0; i < params.size; ++i) {
    const SquareSumParam<T> param = params.params[i];
    CUDA_1D_KERNEL_LOOP(j, param.count) { t_sum += param.ptr[j] * param.ptr[j]; }
  }
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T b_sum = BlockReduce(temp_storage).Sum(t_sum);
  if (threadIdx.x == 0) { cuda::atomic::Add(y, b_sum); }
}

}  // namespace

template<typename T>
struct SquareSumKernelUtil<DeviceType::kGPU, T> {
  static void SquareSum(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    const int32_t num_blocks = BlocksNum4ThreadsNum(n);
    CHECK_GE(num_blocks, 0);
    if (num_blocks == 0) {
      Memset<DeviceType::kGPU>(ctx, y, 0, sizeof(T));
    } else if (num_blocks == 1) {
      SquareSumGpu<T, true><<<1, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
    } else {
      Memset<DeviceType::kGPU>(ctx, y, 0, sizeof(T));
      SquareSumGpu<T, false>
          <<<num_blocks, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
    }
  }

  static void MultiSquareSum(DeviceCtx* ctx, const std::vector<SquareSumParam<T>>& params, T* y) {
    Memset<DeviceType::kGPU>(ctx, y, 0, sizeof(T));
    for (int64_t start = 0; start < params.size(); start += kMultiSquareSumMaxSize) {
      MultiSquareSumParams<T> gpu_params{};
      int64_t max_count = 0;
      gpu_params.size = std::min<int64_t>(start + kMultiSquareSumMaxSize, params.size()) - start;
      for (int64_t i = 0; i < gpu_params.size; ++i) {
        gpu_params.params[i] = params[start + i];
        max_count = std::max(max_count, gpu_params.params[i].count);
      }
      MultiSquareSumGpu<T>
          <<<BlocksNum4ThreadsNum(max_count), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              gpu_params, y);
    }
  }
};

#define INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU(type_cpp, type_proto) \
  template struct SquareSumKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU

}  // namespace oneflow
