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

__device__ int8_t GenMask(curandState* state, const float rate) {
  return curand_uniform(state) >= rate;
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

void RandomMaskGenerator<DeviceType::kGPU>::Generate(DeviceCtx* device_ctx, const int64_t n,
                                                     const float rate, int8_t* mask) {
  int32_t block_num = generator_->max_block_num();
  int32_t thread_num = generator_->max_thread_num();
  auto* curand_states = generator_->curand_states();
  const int32_t elem_cnt_per_block = thread_num * sizeof(PackType) * kMinPackPerThread;
  const int32_t block_num_final =
      std::min(static_cast<int32_t>((n + elem_cnt_per_block - 1) / elem_cnt_per_block), block_num);
  GenerateGpu<<<block_num_final, thread_num, 0, device_ctx->cuda_stream()>>>(curand_states, n, rate,
                                                                             mask);
}

template class RandomMaskGenerator<DeviceType::kGPU>;

}  // namespace oneflow
