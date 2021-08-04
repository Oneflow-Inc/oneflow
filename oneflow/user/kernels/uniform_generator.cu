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
#include "oneflow/user/kernels/uniform_generator.h"
#include "oneflow/core/common/data_type.h"

// TODO(bowenc): support int uniform and uniform with range

namespace oneflow {

namespace {

constexpr int32_t kMinPackPerThread = 2;

using PackType = ulonglong2;

union Pack {
  PackType p_value;
  int8_t b_value[sizeof(PackType)];
};

template<typename T>
__device__ T GenUniform(curandState* state);

template<>
__device__ float GenUniform<float>(curandState* state) {
  return curand_uniform(state);
}

template<>
__device__ double GenUniform<double>(curandState* state) {
  return curand_uniform_double(state);
}

template<typename T>
__global__ void GenerateGpu(curandState* state, const int64_t elem_cnt, T* dptr) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = state[id];
  PackType* pack_mask = reinterpret_cast<PackType*>(dptr);
  Pack pack;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt / sizeof(PackType)) {
#pragma unroll
    for (int j = 0; j < sizeof(PackType); ++j) { pack.b_value[j] = GenUniform<T>(&localState); }
    pack_mask[i] = pack.p_value;
  }
  const int32_t rem_cnt = elem_cnt % sizeof(PackType);
  const int32_t rem_offset = elem_cnt - rem_cnt;
  if (id < rem_cnt) { dptr[id + rem_offset] = GenUniform<T>(&localState); }
  state[id] = localState;
}

}  // namespace

template<typename T>
void UniformGenerator<DeviceType::kGPU>::Generate(DeviceCtx* device_ctx, const int64_t elem_cnt,
                                                  T* dptr) {
  int32_t block_num = generator_->max_block_num();
  int32_t thread_num = generator_->max_thread_num();
  auto* curand_states = generator_->curand_states();
  const int32_t elem_cnt_per_block = thread_num * sizeof(PackType) * kMinPackPerThread;
  const int32_t block_num_final = std::min(
      static_cast<int32_t>((elem_cnt + elem_cnt_per_block - 1) / elem_cnt_per_block), block_num);
  GenerateGpu<T><<<block_num_final, thread_num, 0, device_ctx->cuda_stream()>>>(curand_states,
                                                                                elem_cnt, dptr);
}

#define INITIATE_GPU_UNIFORM_GENERATOR(T, typeproto)                                    \
  template void UniformGenerator<DeviceType::kGPU>::Generate<T>(DeviceCtx * device_ctx, \
                                                                const int64_t elem_cnt, T* dptr);

OF_PP_FOR_EACH_TUPLE(INITIATE_GPU_UNIFORM_GENERATOR, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
