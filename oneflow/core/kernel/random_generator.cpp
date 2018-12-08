#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename T>
void RandomGenerator<DeviceType::kCPU>::Uniform(const int64_t elem_cnt, T* dptr) {
  Uniform(elem_cnt, ZeroVal<T>::value, OneVal<T>::value, dptr);
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
