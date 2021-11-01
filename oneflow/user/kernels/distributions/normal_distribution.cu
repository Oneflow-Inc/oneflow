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

namespace oneflow {

namespace {

template<typename T>
__device__ T GenNormal(curandState* state, const T mean, const T std);

template<>
__device__ float GenNormal<float>(curandState* state, const float mean, const float std) {
  return (curand_normal(state) + mean) / std;
}

template<>
__device__ double GenNormal<double>(curandState* state, const double mean, const double std) {
  return (curand_normal_double(state) + mean) / std;
}

template<typename T>
__global__ void GenerateGpu(curandState* state, const int64_t elem_cnt, T* dptr, const T mean,
                            const T std) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = state[id];
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { dptr[i] = GenNormal<T>(&localState, mean, std); }
  state[id] = localState;
}

}  // namespace

template<typename T>
void NormalDistribution<DeviceType::kGPU, T>::operator()(
    DeviceCtx* device_ctx, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  auto gen = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
  int32_t block_num = gen->max_block_num();
  int32_t thread_num = gen->max_thread_num();
  auto* curand_states = gen->curand_states();
  GenerateGpu<T><<<block_num, thread_num, 0, device_ctx->cuda_stream()>>>(curand_states, elem_cnt,
                                                                          dptr, mean_, std_);
}

#define INITIATE_GPU_NORMAL_DISTRIBUTION(T, typeproto)               \
  template void NormalDistribution<DeviceType::kGPU, T>::operator()( \
      DeviceCtx* device_ctx, const int64_t elem_cnt, T* dptr,        \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_GPU_NORMAL_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
