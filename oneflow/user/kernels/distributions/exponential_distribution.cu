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

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/distributions/distribution_template_util.cuh"
#include "oneflow/user/kernels/distributions/exponential_distribution.h"
#include "oneflow/user/kernels/fused_rnn_cell_kernel_util.h"

namespace oneflow {

namespace {

OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel_double(int32_t numel, uint64_t seed,
                                                                   uint64_t offset, double lambd,
                                                                   double epsilon,
                                                                   double* out_ptr) {
  int32_t unroll_factor = 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x
                     * gridDim.x * unroll_factor;
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    double2 rand = curand_uniform2_double(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        double log_rand = ::log(static_cast<double>((&rand.x)[ii]));
        // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
        // we need log to be not 0, and not underflow when converted to half
        // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1
        // args
        double log = static_cast<double>((&rand.x)[ii]) >= static_cast<double>(1.) - epsilon / 2
                         ? -epsilon / 2
                         : log_rand;
        out_ptr[li] = static_cast<double>(-1.0) / lambd * log;
      }
    }
    __syncthreads();
  }
}

OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel_float(int32_t numel, uint64_t seed,
                                                                  uint64_t offset, float lambd,
                                                                  float epsilon, float* out_ptr) {
  int32_t unroll_factor = 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x
                     * gridDim.x * unroll_factor;
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    float4 rand = curand_uniform4(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        float log_rand = __logf(static_cast<float>((&rand.x)[ii]));
        // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
        // we need log to be not 0, and not underflow when converted to half
        // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1
        // args
        float log = static_cast<float>((&rand.x)[ii]) >= static_cast<float>(1.) - epsilon / 2
                        ? -epsilon / 2
                        : log_rand;
        out_ptr[li] = static_cast<float>(-1.0) / lambd * log;
      }
    }
    __syncthreads();
  }
}

}  // namespace

template<>
void ExponentialDistribution<DeviceType::kCUDA, double>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, double* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GT(elem_cnt, 0);
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

  // TODO(binbin): calling a constexpr __host__ function from a __device__ function is not allowed.
  // The experimental flag '--expt-relaxed-constexpr' of nvcc can be used. And this experimental
  // feature needs to be further researched.
  double epsilon = std::numeric_limits<double>::epsilon();
  auto transform_func = [=] __device__(double random_val) -> double {
    double log_rand = ::log(static_cast<double>(random_val));
    // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
    // we need log to be not 0, and not underflow when converted to half
    // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1
    // args
    double log = static_cast<double>(random_val) >= static_cast<double>(1.) - epsilon / 2
                     ? -epsilon / 2
                     : log_rand;
    return static_cast<double>(-1.0) / lambd_ * log;
  };

  DistributionElementwiseGridStrideKernel<double, 2>
      <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, seed, offset, dptr,
          [] __device__(curandStatePhilox4_32_10_t * state) {
            return curand_normal2_double(state);
          },
          transform_func);
}

template<>
void ExponentialDistribution<DeviceType::kCUDA, float>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, float* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GT(elem_cnt, 0);
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

  // TODO(binbin): calling a constexpr __host__ function from a __device__ function is not allowed.
  // The experimental flag '--expt-relaxed-constexpr' of nvcc can be used. And this experimental
  // feature needs to be further researched.
  float epsilon = std::numeric_limits<float>::epsilon();
  auto transform_func = [=] __device__(float random_val) -> float {
    float log_rand = __logf(static_cast<float>(random_val));
    // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
    // we need log to be not 0, and not underflow when converted to half
    // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1
    // args
    float log = static_cast<float>(random_val) >= static_cast<float>(1.) - epsilon / 2
                    ? -epsilon / 2
                    : log_rand;
    return static_cast<float>(-1.0) / lambd_ * log;
  };

  DistributionElementwiseGridStrideKernel<float, 4>
      <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, seed, offset, dptr,
          [] __device__(curandStatePhilox4_32_10_t * state) { return curand_normal4(state); },
          transform_func);
}

}  // namespace oneflow
