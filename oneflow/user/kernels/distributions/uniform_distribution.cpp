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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/uniform_distribution.h"

namespace oneflow {

template<typename T>
template<typename Integer = T,
         typename std::enable_if<std::is_integral<Integer>::value, int>::type = 0>
void UniformDistribution<DeviceType::kCPU, T>::operator()(
    DeviceCtx* device_ctx, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  auto gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
  std::uniform_int_distribution<T> random_distribution(low_, high_);
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(gen->engine()); }
}

template<typename T>
template<typename Floating = T,
         typename std::enable_if<std::is_floating_point<Floating>::value, int>::type = 0>
void UniformDistribution<DeviceType::kCPU, T>::operator()(
    DeviceCtx* device_ctx, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  auto gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
  std::uniform_real_distribution<T> random_distribution(low_, high_);
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(gen->engine()); }
}

#define INITIATE_CPU_UNIFORM_DISTRIBUTION(T, typeproto)               \
  template void UniformDistribution<DeviceType::kCPU, T>::operator()( \
      DeviceCtx* device_ctx, const int64_t elem_cnt, T* dptr,         \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)
OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_DISTRIBUTION, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
