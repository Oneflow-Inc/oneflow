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

  DistributionElementwiseGridStrideParams params;
  params.numel = elem_cnt;
  params.seed = seed;
  params.offset = offset;
  params.dst = dptr;
  params.attr0 = Scalar(std::numeric_limits<double>::epsilon());
  params.attr1 = Scalar(lambd_);

  DistributionElementwiseGridStrideKernel<double, double, 2, DistributionOp::kUniform2Double,
                                          TransformOp::kExponential>
      <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(params);
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

  DistributionElementwiseGridStrideParams params;
  params.numel = elem_cnt;
  params.seed = seed;
  params.offset = offset;
  params.dst = dptr;
  params.attr0 = Scalar(std::numeric_limits<float>::epsilon());
  params.attr1 = Scalar(lambd_);

  DistributionElementwiseGridStrideKernel<float, float, 4, DistributionOp::kUniform4,
                                          TransformOp::kExponential>
      <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(params);
}

}  // namespace oneflow
