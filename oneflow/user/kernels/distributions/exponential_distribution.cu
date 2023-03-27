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

template<typename T, typename ComputeType>
struct ExponentialTransformFunctor;

template<>
struct ExponentialTransformFunctor<float, float> {
  ExponentialTransformFunctor(float epsilon, float lambd) : epsilon(epsilon), lambd(lambd) {}
  __device__ float operator()(float random_val) const {
    float log_rand = __logf(static_cast<float>(random_val));
    // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
    // we need log to be not 0, and not underflow when converted to half
    // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1
    // args
    float log = static_cast<float>(random_val) >= static_cast<float>(1.) - epsilon / 2
                    ? -epsilon / 2
                    : log_rand;
    return static_cast<float>(-1.0) / lambd * log;
  }
  float epsilon;
  float lambd;
};

template<>
struct ExponentialTransformFunctor<double, double> {
  ExponentialTransformFunctor(double epsilon, double lambd) : epsilon(epsilon), lambd(lambd) {}
  __device__ double operator()(double random_val) const {
    double log_rand = ::log(static_cast<double>(random_val));
    // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
    // we need log to be not 0, and not underflow when converted to half
    // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1
    // args
    double log = static_cast<double>(random_val) >= static_cast<double>(1.) - epsilon / 2
                     ? -epsilon / 2
                     : log_rand;
    return static_cast<double>(-1.0) / lambd * log;
  }
  double epsilon;
  double lambd;
};

template<>
struct ExponentialTransformFunctor<half, float> {
  ExponentialTransformFunctor(float epsilon, float lambd) : float_functor(epsilon, lambd) {}
  __device__ half operator()(float random_val) const {
    return static_cast<half>(float_functor(random_val));
  }
  ExponentialTransformFunctor<float, float> float_functor;
};

template<>
void ExponentialDistribution<DeviceType::kCUDA, double>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, double* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GT(elem_cnt, 0);
  const auto device_index = stream->device()->device_index();
  auto gen = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));
  ep::CudaStream* cuda_stream = stream->As<ep::CudaStream>();
  auto execution_policy = gen->CalcExecutionPolicy(elem_cnt, cuda_stream);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  uint64_t seed = gen->current_seed();
  uint64_t offset = gen->get_philox_offset(counter_offset);

  ExponentialTransformFunctor<double, double> transform_functor(
      std::numeric_limits<double>::epsilon(), static_cast<double>(lambd_));
  DistributionFunctor<DistributionOp::kUniform2Double> dist_functor;

  DistributionElementwiseGridStrideKernel<double, double, 2, decltype(dist_functor),
                                          decltype(transform_functor)>
      <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, seed, offset, dptr, dist_functor, transform_functor);
}

template<>
void ExponentialDistribution<DeviceType::kCUDA, float>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, float* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GT(elem_cnt, 0);
  const auto device_index = stream->device()->device_index();
  auto gen = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));
  ep::CudaStream* cuda_stream = stream->As<ep::CudaStream>();
  auto execution_policy = gen->CalcExecutionPolicy(elem_cnt, cuda_stream);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  uint64_t seed = gen->current_seed();
  uint64_t offset = gen->get_philox_offset(counter_offset);

  ExponentialTransformFunctor<float, float> transform_functor(std::numeric_limits<float>::epsilon(),
                                                              static_cast<float>(lambd_));
  DistributionFunctor<DistributionOp::kUniform4> dist_functor;

  DistributionElementwiseGridStrideKernel<float, float, 4, decltype(dist_functor),
                                          decltype(transform_functor)>
      <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, seed, offset, dptr, dist_functor, transform_functor);
}

template<>
void ExponentialDistribution<DeviceType::kCUDA, half>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, half* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GT(elem_cnt, 0);
  const auto device_index = stream->device()->device_index();
  auto gen = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));
  ep::CudaStream* cuda_stream = stream->As<ep::CudaStream>();
  auto execution_policy = gen->CalcExecutionPolicy(elem_cnt, cuda_stream);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  uint64_t seed = gen->current_seed();
  uint64_t offset = gen->get_philox_offset(counter_offset);

  ExponentialTransformFunctor<half, float> transform_functor(std::numeric_limits<float>::epsilon(),
                                                             static_cast<float>(lambd_));
  DistributionFunctor<DistributionOp::kUniform4> dist_functor;

  DistributionElementwiseGridStrideKernel<half, float, 4, decltype(dist_functor),
                                          decltype(transform_functor)>
      <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, seed, offset, dptr, dist_functor, transform_functor);
}

}  // namespace oneflow
