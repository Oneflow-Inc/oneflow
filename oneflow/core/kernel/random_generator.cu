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
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
void RngUniformGpu(const curandGenerator_t& gen, int64_t n, T* ret);

template<>
void RngUniformGpu<float>(const curandGenerator_t& gen, int64_t n, float* ret) {
  OF_CURAND_CHECK(curandGenerateUniform(gen, ret, n));
}

template<>
void RngUniformGpu<double>(const curandGenerator_t& gen, int64_t n, double* ret) {
  OF_CURAND_CHECK(curandGenerateUniformDouble(gen, ret, n));
}

}  // namespace

RandomGenerator<DeviceType::kCUDA>::RandomGenerator(int64_t seed, ep::Stream* stream) {
  OF_CURAND_CHECK(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  OF_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_, seed));
  OF_CURAND_CHECK(curandSetStream(curand_generator_, stream->As<ep::CudaStream>()->cuda_stream()));
}

RandomGenerator<DeviceType::kCUDA>::~RandomGenerator() {
  OF_CURAND_CHECK(curandDestroyGenerator(curand_generator_));
}

template<typename T>
void RandomGenerator<DeviceType::kCUDA>::Uniform(const int64_t elem_cnt, T* dptr) {
  RngUniformGpu(curand_generator_, elem_cnt, dptr);
}

#define INITIATE_CUDA_RANDOM_GENERATOR_UNIFORM(T, typeproto) \
  template void RandomGenerator<DeviceType::kCUDA>::Uniform<T>(const int64_t elem_cnt, T* dptr);

OF_PP_FOR_EACH_TUPLE(INITIATE_CUDA_RANDOM_GENERATOR_UNIFORM, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
