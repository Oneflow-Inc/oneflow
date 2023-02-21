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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/distributions/distribution_template_util.cuh"

namespace oneflow {

namespace {

using PackType = ulonglong2;

union Pack {
  PackType p_value;
  bool b_value[sizeof(PackType)];
};

__device__ bool GenMask(curandStatePhilox4_32_10_t* state, const float rate) {
  return curand_uniform(state) > rate;
}

__global__ void GenerateGpu(uint64_t seed, uint64_t offset, const int64_t n, const float rate,
                            bool* mask) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, id, offset, &state);
  PackType* pack_mask = reinterpret_cast<PackType*>(mask);
  Pack pack;
  CUDA_1D_KERNEL_LOOP(i, n / sizeof(PackType)) {
#pragma unroll
    for (int j = 0; j < sizeof(PackType); j += 4) {
      auto rand = curand_uniform4(&state);
      pack.b_value[j] = (&rand.x)[0] > rate;
      pack.b_value[j + 1] = (&rand.x)[1] > rate;
      pack.b_value[j + 2] = (&rand.x)[2] > rate;
      pack.b_value[j + 3] = (&rand.x)[3] > rate;
    }
    pack_mask[i] = pack.p_value;
  }

  const int32_t rem_cnt = n % sizeof(PackType);
  const int32_t rem_offset = n - rem_cnt;
  if (id < rem_cnt) { mask[id + rem_offset] = GenMask(&state, rate); }
}

}  // namespace

void RandomMaskGenerator<DeviceType::kCUDA>::Generate(ep::Stream* stream, const int64_t n,
                                                      const float rate, bool* mask) {
  if (n == 0) return;
  ep::CudaStream* cuda_stream = stream->As<ep::CudaStream>();
  auto execution_policy = generator_->CalcExecutionPolicy(n, cuda_stream);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  uint64_t seed = generator_->current_seed();
  uint64_t offset = generator_->get_philox_offset(counter_offset);

  GenerateGpu<<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(seed, offset, n,
                                                                               rate, mask);
}

template class RandomMaskGenerator<DeviceType::kCUDA>;

}  // namespace oneflow
