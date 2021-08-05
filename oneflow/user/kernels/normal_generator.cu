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
#include "oneflow/user/kernels/normal_generator.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace {

template<typename T>
__device__ T GenNormal(curandState* state);

template<>
__device__ float GenNormal<float>(curandState* state) {
  return curand_normal(state);
}

template<>
__device__ double GenNormal<double>(curandState* state) {
  return curand_normal_double(state);
}

template<typename T>
__global__ void GenerateGpu(curandState* state, const int64_t elem_cnt, T* dptr) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = state[id];
  if (id < elem_cnt) { dptr[id] = GenNormal<T>(&localState); }
  state[id] = localState;
}

}  // namespace

template<typename T>
void NormalGenerator<DeviceType::kGPU>::Generate(DeviceCtx* device_ctx, const int64_t elem_cnt,
                                                  T* dptr) {
  int32_t block_num = generator_->max_block_num();
  int32_t thread_num = generator_->max_thread_num();
  auto* curand_states = generator_->curand_states();
  GenerateGpu<T>
      <<<block_num, thread_num, 0, device_ctx->cuda_stream()>>>(curand_states, elem_cnt, dptr);
}


#define INITIATE_GPU_NORMAL_GENERATOR(T, typeproto)                                    \
  template void NormalGenerator<DeviceType::kGPU>::Generate<T>(DeviceCtx * device_ctx, \
                                                                const int64_t elem_cnt, T* dptr);

OF_PP_FOR_EACH_TUPLE(INITIATE_GPU_NORMAL_GENERATOR, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow