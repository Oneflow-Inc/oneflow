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
#include "hip/hip_runtime.h"
#include "oneflow/core/framework/random_generator_impl.h"

namespace oneflow {
namespace one {

namespace {

__global__ void InitCurandStatesKernel(uint64_t seed, hiprandState* states,
                                       CUDAGeneratorState* cuda_gen_state) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t local_seed = (static_cast<size_t>(seed) + 0x9e3779b9U + (static_cast<size_t>(id) << 6U)
                       + (static_cast<size_t>(id) >> 2U));
  hiprand_init(local_seed, 0, 0, &states[id]);
  cuda_gen_state->dev_counter = static_cast<int32_t>(0);
  cuda_gen_state->dev_offset = static_cast<uint64_t>(0);
}

}  // namespace

namespace detail {

void InitCurandStates(uint64_t seed, int32_t block_num, int32_t thread_num, hiprandState* states,
                      CUDAGeneratorState* cuda_gen_state) {
  hipLaunchKernelGGL(InitCurandStatesKernel, block_num, thread_num, 0, 0, seed, states, cuda_gen_state);
}

}  // namespace detail

}  // namespace one
}  // namespace oneflow