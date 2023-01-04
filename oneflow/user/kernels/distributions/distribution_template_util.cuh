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
#ifndef ONEFLOW_USER_KERNELS_DISTRIBUTIONS__DISTRIBUTIONS_TEMPLATE_UTIL_H_
#define ONEFLOW_USER_KERNELS_DISTRIBUTIONS__DISTRIBUTIONS_TEMPLATE_UTIL_H_

#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/fused_rnn_cell_kernel_util.h"
#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace oneflow {

namespace {

// launch bounds used for kernels
const uint32_t block_size_bound = 256;
const uint32_t grid_size_bound = 4;
// number of randoms given by distributions like curand_uniform4, curand_uniform2_double
// used in calculating philox offset.
const uint32_t curand4_engine_calls = 4;

std::tuple<uint64_t, dim3, dim3> CalcExecutionPolicy(int64_t total_elements,
                                                     ep::CudaStream* stream) {
  // NOTE(Liang Depeng): the implementation is modified from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/DistributionTemplates.h

  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = block_size_bound;
  const uint32_t unroll = curand4_engine_calls;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  uint32_t blocks_per_sm = stream->device_properties().maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(stream->device_properties().multiProcessorCount) * blocks_per_sm,
      grid.x);
  // number of times random will be generated per thread, to offset philox counter in thc random
  // state
  uint64_t counter_offset =
      ((numel - 1) / (block_size * grid.x * unroll) + 1) * curand4_engine_calls;
  return std::make_tuple(counter_offset, grid, dim_block);
}

}  // namespace

#ifdef WITH_CUDA

template<typename T, int unroll_factor, typename dist_t, typename transform_t>
OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__
    void DistributionElementwiseGridStrideKernel(int32_t numel, uint64_t seed, uint64_t offset,
                                                 T* out_ptr, const dist_t& dist_func,
                                                 const transform_t transform_func) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x
                     * gridDim.x * unroll_factor;
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_func(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) { out_ptr[li] = transform_func((&rand.x)[ii]); }
    }
    __syncthreads();
  }
}

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DISTRIBUTIONS__DISTRIBUTIONS_TEMPLATE_UTIL_H_
