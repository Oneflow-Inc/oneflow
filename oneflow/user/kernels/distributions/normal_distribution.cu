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

#include "oneflow/user/kernels/distributions/normal_distribution.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/fused_rnn_cell_kernel_util.h"
#include "oneflow/core/cuda/layer_norm.cuh"

namespace oneflow {

namespace {

// launch bounds used for kernels
const uint32_t block_size_bound = 256;
const uint32_t grid_size_bound = 4;

std::tuple<uint64_t, dim3, dim3> CalcExecutionPolicy(int64_t total_elements,
                                                     ep::CudaStream* stream) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = block_size_bound;
  // number of randoms given by distributions like curand_uniform4, curand_uniform2_double
  // used in calculating philox offset.
  const uint32_t curand4_engine_calls = 4;
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

template<typename T, typename ComputeType, int unroll_factor>
OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void DistributionElementwiseGridStrideKernelDouble(int32_t numel, uint64_t seed,
                                                              uint64_t offset, ComputeType mean,
                                                              ComputeType std, T* out_ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x
                     * gridDim.x * unroll_factor;
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    double2 rand = curand_normal2_double(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        out_ptr[li] = static_cast<T>(static_cast<ComputeType>((&rand.x)[ii]) * std + mean);
      }
    }
    __syncthreads();
  }
}

template<typename T, typename ComputeType, int unroll_factor>
OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void DistributionElementwiseGridStrideKernelFloat(int32_t numel, uint64_t seed,
                                                             uint64_t offset, ComputeType mean,
                                                             ComputeType std, T* out_ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x
                     * gridDim.x * unroll_factor;
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    float4 rand = curand_normal4(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        out_ptr[li] = static_cast<T>(static_cast<ComputeType>((&rand.x)[ii]) * std + mean);
      }
    }
    __syncthreads();
  }
}

}  // namespace

template<typename T>
void NormalDistribution<DeviceType::kCUDA, T>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  if (elem_cnt == 0) return;
  const auto device_index = stream->device()->device_index();
  auto gen = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));

  ep::CudaStream* cuda_stream = stream->As<ep::CudaStream>();
  auto execution_policy = CalcExecutionPolicy(elem_cnt, cuda_stream);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  uint64_t offset = 0;
  uint64_t seed = gen->current_seed();
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    offset = gen->get_philox_offset(counter_offset);
  }

  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  if (std::is_same<T, double>::value) {
    DistributionElementwiseGridStrideKernelDouble<T, ComputeType, 2>
        <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, seed, offset, static_cast<ComputeType>(mean_), static_cast<ComputeType>(std_),
            dptr);
  } else {
    DistributionElementwiseGridStrideKernelFloat<T, ComputeType, 4>
        <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, seed, offset, static_cast<ComputeType>(mean_), static_cast<ComputeType>(std_),
            dptr);
  }
}

#define INITIATE_CUDA_NORMAL_DISTRIBUTION(T, typeproto)               \
  template void NormalDistribution<DeviceType::kCUDA, T>::operator()( \
      ep::Stream* stream, const int64_t elem_cnt, T* dptr,            \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CUDA_NORMAL_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)
INITIATE_CUDA_NORMAL_DISTRIBUTION(half, DataType::kFloat16)

}  // namespace oneflow
