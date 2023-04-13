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
#ifndef ONEFLOW_USER_KERNELS_BATCH_NORM_UTILS_H_
#define ONEFLOW_USER_KERNELS_BATCH_NORM_UTILS_H_
// NOTE(Liang Depeng):
// Modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Normalization.cuh

#if defined(__CUDACC__)

constexpr int ELEMENTS_PER_ITER = 4;  // enables concurrency within each thread to hide latency
constexpr int ELEMENTS_PER_THREAD = 16;
constexpr int OPTIMAL_TILE_W = 32;
constexpr int MAX_H_BLOCK = 128;
constexpr int32_t MAX_BLOCK_SIZE = 512;
constexpr unsigned MAX_GRID_SIZE = 65535u;
#define WARP_SIZE 32

// returns 2**floor(log2(n))
static int lastPow2(unsigned int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max<int>(1, n - (n >> 1));
}

/**
   Computes ceil(a / b)
*/
template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
static T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

static void flexible_launch_configs(const int reduction, const int stride, dim3& block, dim3& grid,
                                    const bool coop_flag = false) {
  int block_x = std::min(lastPow2(stride), OPTIMAL_TILE_W);
  int block_y =
      std::min(lastPow2(ceil_div(reduction, ELEMENTS_PER_THREAD)), MAX_BLOCK_SIZE / block_x);
  if (block_x * block_y != MAX_BLOCK_SIZE) {
    block_x = std::min(lastPow2(stride), MAX_BLOCK_SIZE / block_y);
  }

  int grid_x = ceil_div(stride, block_x);
  int grid_y = std::min(ceil_div(reduction, block_y * ELEMENTS_PER_THREAD), MAX_H_BLOCK);
  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  block.x = block_x;
  block.y = block_y;
  block.z = 1;
  grid.x = grid_x;
  grid.y = grid_y;
  grid.z = 1;
}

template<typename T>
struct AccumulateType {};
template<>
struct AccumulateType<float> {
  using type = float;
};
template<>
struct AccumulateType<double> {
  using type = double;
};

template<typename T>
using acc_type = typename AccumulateType<T>::type;

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int32_t getNumThreads(int64_t nElem) {
  int32_t threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
  for (int32_t i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) { return threadSizes[i]; }
  }
  return MAX_BLOCK_SIZE;
}

template<typename T>
static __forceinline__ __device__ T device_sqrt(T val);

template<>
__forceinline__ __device__ float device_sqrt(float val) {
  return ::sqrtf(val);
}

template<>
__forceinline__ __device__ double device_sqrt(double val) {
  return ::sqrt(val);
}

template<typename T>
__device__ __forceinline__ T inv_std(T var, double eps) {
  T invstd = 0;
  if (var != static_cast<T>(0) || eps != static_cast<T>(0)) {
    invstd = static_cast<T>(1) / device_sqrt(var + eps);
  }
  return invstd;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int32_t getMSB(int32_t val) { return 31 - __clz(val); }

template<typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize,
                                           unsigned int mask = 0xffffffff) {
  return __shfl_xor_sync(mask, value, laneMask, width);
}

#endif

#endif  // ONEFLOW_USER_KERNELS_BATCH_NORM_UTILS_H_
