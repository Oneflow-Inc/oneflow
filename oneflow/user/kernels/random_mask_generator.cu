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

void RandomMaskGenerator<DeviceType::kCUDA>::Generate(ep::Stream* stream, const int64_t n,
                                                      const float rate, bool* mask) {
  if (n == 0) return;
  ep::CudaStream* cuda_stream = stream->As<ep::CudaStream>();
  auto execution_policy = CalcExecutionPolicy(n, cuda_stream);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  uint64_t offset = 0;
  uint64_t seed = generator_->current_seed();
  {
    std::lock_guard<std::mutex> lock(generator_->mutex_);
    offset = generator_->get_philox_offset(counter_offset);
  }

  auto transform_func = [=] __device__(float rand_val) -> bool { return rand_val > rate; };

  DistributionElementwiseGridStrideKernel<bool, 4>
      <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          n, seed, offset, mask,
          [] __device__(curandStatePhilox4_32_10_t * state) { return curand_uniform4(state); },
          transform_func);
}

template class RandomMaskGenerator<DeviceType::kCUDA>;

}  // namespace oneflow
