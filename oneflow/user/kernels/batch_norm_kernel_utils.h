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
#if defined(__CUDACC__)

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

constexpr int32_t MAX_BLOCK_SIZE = 512;
constexpr unsigned MAX_GRID_SIZE = 65535u;
#define WARP_SIZE 32

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
