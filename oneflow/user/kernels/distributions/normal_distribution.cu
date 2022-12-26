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

namespace oneflow {

namespace {

template<typename T>
struct AccumulateType {};
template<>
struct AccumulateType<float> {
  using type = float;
};
template<>
struct AccumulateType<double> {
  using type = double;
};

template <>
struct AccumulateType<half> {
  using type = float;
};

template<typename T>
using acc_type = typename AccumulateType<T>::type;

// launch bounds used for kernels
const uint32_t block_size_bound = 256;
const uint32_t grid_size_bound = 4;
// number of randoms given by distributions like curand_uniform4, curand_uniform2_double
// used in calculating philox offset.
const uint32_t curand4_engine_calls = 4;

std::tuple<uint64_t, dim3, dim3> calc_execution_policy(int64_t total_elements,
                                                       ep::CudaStream* stream) {
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


template<typename T>
__device__ T GenNormal(curandState* state, const T mean, const T std);

template<>
__device__ float GenNormal<float>(curandState* state, const float mean, const float std) {
  return curand_normal(state) * std + mean;
}

template<>
__device__ double GenNormal<double>(curandState* state, const double mean, const double std) {
  return curand_normal_double(state) * std + mean;
}

template<typename T>
__global__ void GenerateGpu(curandState* state, const int64_t elem_cnt, T* dptr, const T mean,
                            const T std) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = state[id];
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { dptr[i] = GenNormal<T>(&localState, mean, std); }
  state[id] = localState;
}

// specialization for half
template<>
__global__ void GenerateGpu(curandState* state, const int64_t elem_cnt, half* dptr, const half mean,
                            const half std) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = state[id];
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    dptr[i] = static_cast<half>(GenNormal<float>(&localState, mean, std));
  }
  state[id] = localState;
}


template<typename T, typename ACC_T>
OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel_double(int32_t numel, uint64_t seed,
                                                                   uint64_t offset, ACC_T mean,
                                                                   ACC_T std, T* out_ptr) {
  int32_t unroll_factor = 2;
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
        out_ptr[li] = static_cast<T>(static_cast<ACC_T>((&rand.x)[ii]) * std + mean);
      }
    }
    __syncthreads();
  }
}

template<typename T, typename ACC_T>
OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel_float(int32_t numel, uint64_t seed,
                                                                  uint64_t offset,  ACC_T mean,
                                                                  ACC_T std, T* out_ptr) {
  int32_t unroll_factor = 4;
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
        out_ptr[li] = static_cast<T>(static_cast<ACC_T>((&rand.x)[ii]) * std + mean);
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
  auto execution_policy = calc_execution_policy(elem_cnt, cuda_stream);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  uint64_t offset = 0;
  uint64_t seed = gen->current_seed();
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    offset = gen->get_philox_offset(counter_offset);
  }

  using ACC_T = acc_type<T>;
  if (std::is_same<T, double>::value) {
    distribution_elementwise_grid_stride_kernel_double<T, ACC_T><<<
      grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        elem_cnt, seed, offset, static_cast<ACC_T>(mean_), static_cast<ACC_T>(std_), dptr);
  } else {
    distribution_elementwise_grid_stride_kernel_float<T, ACC_T><<<
      grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        elem_cnt, seed, offset, static_cast<ACC_T>(mean_), static_cast<ACC_T>(std_), dptr);
  }

  // int32_t block_num = gen->max_block_num();
  // int32_t thread_num = gen->max_thread_num();
  // auto* curand_states = gen->curand_states();
  // GenerateGpu<T><<<block_num, thread_num, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
  //     curand_states, elem_cnt, dptr, mean_, std_);
}

#define INITIATE_CUDA_NORMAL_DISTRIBUTION(T, typeproto)               \
  template void NormalDistribution<DeviceType::kCUDA, T>::operator()( \
      ep::Stream* stream, const int64_t elem_cnt, T* dptr,            \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CUDA_NORMAL_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)
INITIATE_CUDA_NORMAL_DISTRIBUTION(half, DataType::kFloat16)

}  // namespace oneflow
