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

#include "oneflow/user/kernels/distributions/distribution_template_util.cuh"
#include "oneflow/user/kernels/distributions/normal_distribution.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/device.h"

namespace oneflow {

template<typename T, typename ComputeType>
struct NormalTransformFunctor {
  NormalTransformFunctor(ComputeType mean, ComputeType std) : mean(mean), std(std) {}
  __device__ T operator()(ComputeType random_val) const {
    return static_cast<T>(random_val * std + mean);
  }
  ComputeType mean;
  ComputeType std;
};

template<typename T>
void NormalDistribution<DeviceType::kCUDA, T>::operator()(
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

  NormalTransformFunctor<T, ComputeType> transform_functor(static_cast<ComputeType>(mean_),
                                                           static_cast<ComputeType>(std_));

  if (std::is_same<T, double>::value) {
    DistributionFunctor<DistributionOp::kNormal2Double> dist_functor;
    DistributionElementwiseGridStrideKernel<T, ComputeType, 2, decltype(dist_functor),
                                            decltype(transform_functor)>
        <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, seed, offset, dptr, dist_functor, transform_functor);
  } else {
    DistributionFunctor<DistributionOp::kNormal4> dist_functor;
    DistributionElementwiseGridStrideKernel<T, ComputeType, 4, decltype(dist_functor),
                                            decltype(transform_functor)>
        <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, seed, offset, dptr, dist_functor, transform_functor);
  }
}

#define INITIATE_CUDA_NORMAL_DISTRIBUTION(T, typeproto)               \
  template void NormalDistribution<DeviceType::kCUDA, T>::operator()( \
      ep::Stream* stream, const int64_t elem_cnt, T* dptr,            \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CUDA_NORMAL_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)
INITIATE_CUDA_NORMAL_DISTRIBUTION(half, DataType::kFloat16)

}  // namespace oneflow
