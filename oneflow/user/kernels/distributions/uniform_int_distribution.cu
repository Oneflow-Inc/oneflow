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
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/user/kernels/distributions/uniform_int_distribution.h"
#include "oneflow/user/kernels/distributions/distribution_template_util.cuh"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

template<typename T, typename ComputeType>
struct UniformIntTransformFunctor {
  UniformIntTransformFunctor(ComputeType low, ComputeType high) : low(low), high(high) {}
  __device__ T operator()(ComputeType rand_num) const {
    if (rand_num == 1.0) { rand_num = 0.0; }
    return static_cast<T>(static_cast<int64_t>(rand_num * (high - low) + low));
  }
  ComputeType low;
  ComputeType high;
};

template<typename T>
void UniformIntDistribution<DeviceType::kCUDA, T>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  if (elem_cnt == 0) return;
  const auto device_index = stream->device()->device_index();
  auto gen = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));

  ep::CudaStream* cuda_stream = stream->As<ep::CudaStream>();
  auto execution_policy = gen->CalcExecutionPolicy(elem_cnt, cuda_stream);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  uint64_t seed = gen->current_seed();
  uint64_t offset = gen->get_philox_offset(counter_offset);

  using ComputeType = typename distribution::DefaultComputeType<T>::type;

  UniformIntTransformFunctor<T, ComputeType> transform_functor(low_, high_);

  if (std::is_same<T, double>::value) {
    DistributionFunctor<DistributionOp::kUniform2Double> dist_functor;
    DistributionElementwiseGridStrideKernel<T, ComputeType, 2, decltype(dist_functor),
                                            decltype(transform_functor)>
        <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, seed, offset, dptr, dist_functor, transform_functor);
  } else {
    DistributionFunctor<DistributionOp::kUniform4> dist_functor;
    DistributionElementwiseGridStrideKernel<T, ComputeType, 4, decltype(dist_functor),
                                            decltype(transform_functor)>
        <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, seed, offset, dptr, dist_functor, transform_functor);
  }
}

#define INITIATE_CUDA_UNIFORM_INT_DISTRIBUTION(T, typeproto)              \
  template void UniformIntDistribution<DeviceType::kCUDA, T>::operator()( \
      ep::Stream* stream, const int64_t elem_cnt, T* dptr,                \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CUDA_UNIFORM_INT_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)
OF_PP_FOR_EACH_TUPLE(INITIATE_CUDA_UNIFORM_INT_DISTRIBUTION, INT_DATA_TYPE_SEQ)
OF_PP_FOR_EACH_TUPLE(INITIATE_CUDA_UNIFORM_INT_DISTRIBUTION, UNSIGNED_INT_DATA_TYPE_SEQ)

}  // namespace oneflow
