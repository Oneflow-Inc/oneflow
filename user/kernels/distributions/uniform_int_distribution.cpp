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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/user/kernels/distributions/uniform_int_distribution.h"

namespace oneflow {

template<typename T>
class CPUUniformIntDistributionImpl {
 public:
  CPUUniformIntDistributionImpl(int64_t low, int64_t high) : random_distribution_(low, high) {}

  T operator()(std::mt19937& engine) { return static_cast<T>(random_distribution_(engine)); }

 private:
  std::uniform_int_distribution<int64_t> random_distribution_;
};

template<typename T>
void UniformIntDistribution<DeviceType::kCPU, T>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  auto gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
  // std::uniform_int_distribution generates [low, high], but we want [low, high) here
  CPUUniformIntDistributionImpl<T> impl(low_, high_ - 1);
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = impl(gen->engine()); }
}

#define INITIATE_CPU_UNIFORM_INT_DISTRIBUTION(T, typeproto)              \
  template void UniformIntDistribution<DeviceType::kCPU, T>::operator()( \
      ep::Stream* stream, const int64_t elem_cnt, T* dptr,               \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_INT_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)
OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_INT_DISTRIBUTION, INT_DATA_TYPE_SEQ)
OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_INT_DISTRIBUTION, UNSIGNED_INT_DATA_TYPE_SEQ)

}  // namespace oneflow
