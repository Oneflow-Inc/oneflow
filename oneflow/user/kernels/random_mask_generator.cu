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
#include "oneflow/user/kernels/random_mask_generator.h"

namespace oneflow {

namespace {

constexpr int32_t kMinPackPerThread = 2;

using PackType = ulonglong2;

union Pack {
  PackType p_value;
  int8_t b_value[sizeof(PackType)];
};

int GetThreadNum(const cudaDeviceProp& prop) {
  switch (prop.major) {
    case 3:  // Kepler
      return 2 * 192;
    case 5:  // Maxwell
      return 2 * 128;
    case 6:  // Pascal
      if ((prop.minor == 1) || (prop.minor == 2)) {
        return 2 * 128;
      } else {
        return 2 * 64;
      }
    case 7:  // Volta and Turing
      return 2 * 64;
    default: return 2 * 64;
  }
}

__device__ int8_t GenMask(curandState* state, const float rate) {
  return curand_uniform(state) >= rate;
}

__global__ void SetupKernel(int64_t seed, curandState* state) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t local_seed = (static_cast<size_t>(seed) + 0x9e3779b9U + (static_cast<size_t>(id) << 6U)
                       + (static_cast<size_t>(id) >> 2U));
  curand_init(local_seed, 0, 0, &state[id]);
}

__global__ void GenerateGpu(curandState* state, const int64_t n, const float rate, int8_t* mask) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = state[id];
  PackType* pack_mask = reinterpret_cast<PackType*>(mask);
  Pack pack;
  CUDA_1D_KERNEL_LOOP(i, n / sizeof(PackType)) {
#pragma unroll
    for (int j = 0; j < sizeof(PackType); ++j) { pack.b_value[j] = GenMask(&localState, rate); }
    pack_mask[i] = pack.p_value;
  }
  const int32_t rem_cnt = n % sizeof(PackType);
  const int32_t rem_offset = n - rem_cnt;
  if (id < rem_cnt) { mask[id + rem_offset] = GenMask(&localState, rate); }
  state[id] = localState;
}

}  // namespace

RandomMaskGenerator<DeviceType::kGPU>::RandomMaskGenerator(int64_t seed) {
  cudaDeviceProp prop;
  OF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  block_num_ = prop.multiProcessorCount;
  thread_num_ = GetThreadNum(prop);
  OF_CUDA_CHECK(cudaMalloc(&curand_states_, block_num_ * thread_num_ * sizeof(curandState)));
  SetupKernel<<<block_num_, thread_num_>>>(seed, curand_states_);
}

RandomMaskGenerator<DeviceType::kGPU>::~RandomMaskGenerator() {
  OF_CUDA_CHECK(cudaFree(curand_states_));
}

void RandomMaskGenerator<DeviceType::kGPU>::Generate(DeviceCtx* device_ctx, const int64_t n,
                                                     const float rate, int8_t* mask) {
  const int32_t elem_cnt_per_block = thread_num_ * sizeof(PackType) * kMinPackPerThread;
  const int32_t block_num =
      std::min(static_cast<int32_t>((n + elem_cnt_per_block - 1) / elem_cnt_per_block), block_num_);
  GenerateGpu<<<block_num, thread_num_, 0, device_ctx->cuda_stream()>>>(curand_states_, n, rate,
                                                                        mask);
}

template class RandomMaskGenerator<DeviceType::kGPU>;

}  // namespace oneflow
