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
  DistributionFunctor() {}

  __device__ float4 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_normal4(state);
  }
};

template<>
struct DistributionFunctor<DistributionOp::kNormal2Double> {
  DistributionFunctor() {}

  __device__ double2 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_normal2_double(state);
  }
};

template<>
struct DistributionFunctor<DistributionOp::kUniform4> {
  DistributionFunctor() {}

  __device__ float4 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_uniform4(state);
  }
};

template<>
struct DistributionFunctor<DistributionOp::kUniform2Double> {
  DistributionFunctor() {}

  __device__ double2 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_uniform2_double(state);
  }
};

template<typename T, typename ComputeType, int unroll_factor, typename Distribution,
         typename Transform>
OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__
    void DistributionElementwiseGridStrideKernel(int64_t numel, uint64_t seed, uint64_t offset,
                                                 T* out_ptr, Distribution dist_func,
                                                 Transform transform_func) {
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
      if (li < numel) { out_ptr[li] = transform_func(static_cast<ComputeType>((&rand.x)[ii])); }
    }
  }
}

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DISTRIBUTIONS_DISTRIBUTIONS_TEMPLATE_UTIL_H_
