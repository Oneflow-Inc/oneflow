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

#include "oneflow/core/framework/random_generator.h"

namespace oneflow {
namespace one {

namespace {

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

__global__ void SetupKernel(uint64_t seed, curandState* state) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t local_seed = (static_cast<size_t>(seed) + 0x9e3779b9U + (static_cast<size_t>(id) << 6U)
                       + (static_cast<size_t>(id) >> 2U));
  curand_init(local_seed, 0, 0, &state[id]);
}

}  // namespace

void DeviceGeneratorImpl<DeviceType::kGPU>::CudaRandInit(uint64_t seed) {
  SetupKernel<<<block_num_, thread_num_>>>(seed, curand_states_);
}

DeviceGeneratorImpl<DeviceType::kGPU>::DeviceGeneratorImpl(uint64_t seed)
    : GeneratorImpl(seed, "cuda") {
  cudaDeviceProp prop;
  OF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  block_num_ = prop.multiProcessorCount;
  thread_num_ = GetThreadNum(prop);
  OF_CUDA_CHECK(cudaMalloc(&curand_states_, block_num_ * thread_num_ * sizeof(curandState)));
  CudaRandInit(seed);
}

DeviceGeneratorImpl<DeviceType::kGPU>::~DeviceGeneratorImpl() {
  OF_CUDA_CHECK(cudaFree(curand_states_));
}

template class DeviceGeneratorImpl<DeviceType::kGPU>;

}  // namespace one
}  // namespace oneflow
