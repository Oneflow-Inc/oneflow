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
#ifndef ONEFLOW_USER_KERNELS_DISTRIBUTIONS_DISTRIBUTIONS_TEMPLATE_UTIL_H_
#define ONEFLOW_USER_KERNELS_DISTRIBUTIONS_DISTRIBUTIONS_TEMPLATE_UTIL_H_

#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/fused_rnn_cell_kernel_util.h"
#include "oneflow/core/common/scalar.h"
#ifdef WITH_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace oneflow {

namespace distribution {

template<typename T>
struct DefaultComputeType {
  using type = T;
};

#define OF_DEINFE_SPECIAL_DEFAULT_COMPUTE_TYPE(T, typeproto) \
  template<>                                                 \
  struct DefaultComputeType<T> {                             \
    using type = float;                                      \
  };

OF_PP_FOR_EACH_TUPLE(OF_DEINFE_SPECIAL_DEFAULT_COMPUTE_TYPE,
                     INT_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
                         HALF_DATA_TYPE_SEQ)

#undef OF_DEINFE_SPECIAL_DEFAULT_COMPUTE_TYPE

}  // namespace distribution

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

enum class DistributionOp {
  kNormal4,
  kNormal2Double,
  kUniform4,
  kUniform2Double,
};

template<DistributionOp distribution_op>
struct DistributionFunctor;

template<>
struct DistributionFunctor<DistributionOp::kNormal4> {
  __device__ __forceinline__ DistributionFunctor() {}

  __device__ __forceinline__ float4 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_normal4(state);
  }
};

template<>
struct DistributionFunctor<DistributionOp::kNormal2Double> {
  __device__ __forceinline__ DistributionFunctor() {}

  __device__ __forceinline__ double2 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_normal2_double(state);
  }
};

template<>
struct DistributionFunctor<DistributionOp::kUniform4> {
  __device__ __forceinline__ DistributionFunctor() {}

  __device__ __forceinline__ float4 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_uniform4(state);
  }
};

template<>
struct DistributionFunctor<DistributionOp::kUniform2Double> {
  __device__ __forceinline__ DistributionFunctor() {}

  __device__ __forceinline__ double2 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_uniform2_double(state);
  }
};

enum class TransformOp {
  kNormal,
  kExponential,
  kUniform,
  kUniformInt,
};

template<TransformOp transform_op, typename T, typename ComputeType>
struct TransformFunctor;

template<typename T, typename ComputeType>
struct TransformFunctor<TransformOp::kNormal, T, ComputeType> {
  __device__ __forceinline__ TransformFunctor(ComputeType mean, ComputeType std) {
    this->mean = mean;
    this->std = std;
  }
  __device__ __forceinline__ T operator()(ComputeType random_val) const {
    return static_cast<T>(random_val * std + mean);
  }
  ComputeType mean;
  ComputeType std;
};

template<typename T, typename ComputeType>
struct TransformFunctor<TransformOp::kExponential, T, ComputeType> {
  __device__ __forceinline__ TransformFunctor(ComputeType epsilon, ComputeType lambd) {
    this->epsilon = epsilon;
    this->lambd = lambd;
  }
  __device__ __forceinline__ T operator()(ComputeType random_val) const {
    ComputeType log_rand = ::log(static_cast<ComputeType>(random_val));
    // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
    // we need log to be not 0, and not underflow when converted to half
    // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1
    // args
    ComputeType log =
        static_cast<ComputeType>(random_val) >= static_cast<ComputeType>(1.) - epsilon / 2
            ? -epsilon / 2
            : log_rand;
    return static_cast<ComputeType>(-1.0) / lambd * log;
  }
  ComputeType epsilon;
  ComputeType lambd;
};

template<typename T, typename ComputeType>
struct TransformFunctor<TransformOp::kUniform, T, ComputeType> {
  __device__ __forceinline__ TransformFunctor(ComputeType low, ComputeType high) {
    this->low = low;
    this->high = high;
  }
  __device__ __forceinline__ T operator()(ComputeType rand_num) const {
    if (rand_num == static_cast<ComputeType>(1.0)) { rand_num = static_cast<ComputeType>(0.0); }
    return static_cast<T>(rand_num * (high - low) + low);
  }
  ComputeType low;
  ComputeType high;
};

template<typename T, typename ComputeType>
struct TransformFunctor<TransformOp::kUniformInt, T, ComputeType> {
  __device__ __host__ __forceinline__ TransformFunctor(ComputeType low, ComputeType high) {
    this->low = low;
    this->high = high;
  }
  __device__ __forceinline__ T operator()(ComputeType rand_num) const {
    if (rand_num == 1.0) { rand_num = 0.0; }
    return static_cast<T>(static_cast<int64_t>(rand_num * (high - low) + low));
  }
  ComputeType low;
  ComputeType high;
};

struct DistributionElementwiseGridStrideParams {
  int32_t numel;
  uint64_t seed;
  uint64_t offset;
  void* dst{};
  Scalar attr0;
  Scalar attr1;
};

template<typename T, typename ComputeType, int unroll_factor, DistributionOp distribution_op,
         TransformOp transform_op>
OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__
    void DistributionElementwiseGridStrideKernel(DistributionElementwiseGridStrideParams params) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(params.seed, idx, params.offset, &state);

  int rounded_size = ((params.numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1)
                     * blockDim.x * gridDim.x * unroll_factor;
  T* out_ptr = reinterpret_cast<T*>(params.dst);
  DistributionFunctor<distribution_op> dist_functor;
  TransformFunctor<transform_op, T, ComputeType> transform_functor(
      params.attr0.Value<ComputeType>(), params.attr1.Value<ComputeType>());
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_functor(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < params.numel) {
        out_ptr[li] = transform_functor(static_cast<ComputeType>((&rand.x)[ii]));
      }
    }
  }
}

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DISTRIBUTIONS_DISTRIBUTIONS_TEMPLATE_UTIL_H_
