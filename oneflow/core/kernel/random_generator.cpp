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
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename T>
void RandomGenerator<DeviceType::kCPU>::Uniform(const int64_t elem_cnt, T* dptr) {
  Uniform(elem_cnt, GetZeroVal<T>(), GetOneVal<T>(), dptr);
}

template<typename T>
void RandomGenerator<DeviceType::kCPU>::Uniform(const int64_t elem_cnt, const T min, const T max,
                                                T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::uniform_real_distribution<T> random_distribution(min, std::nextafter(max, GetMaxVal<T>()));
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(mt19937_generator_); }
}

#define INITIATE_CPU_RANDOM_GENERATOR_UNIFORM(T, typeproto)                                        \
  template void RandomGenerator<DeviceType::kCPU>::Uniform<T>(const int64_t elem_cnt, T* dptr);    \
  template void RandomGenerator<DeviceType::kCPU>::Uniform<T>(const int64_t elem_cnt, const T min, \
                                                              const T max, T* dptr);

OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_RANDOM_GENERATOR_UNIFORM, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
