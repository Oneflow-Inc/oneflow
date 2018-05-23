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
  std::uniform_real_distribution<T> random_distribution(
      min, std::nextafter(max, std::numeric_limits<T>::max()));
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(mt19937_generator_); }
}

template void RandomGenerator<DeviceType::kCPU>::Uniform<float>(const int64_t elem_cnt,
                                                                float* dptr);
template void RandomGenerator<DeviceType::kCPU>::Uniform<double>(const int64_t elem_cnt,
                                                                 double* dptr);
template void RandomGenerator<DeviceType::kCPU>::Uniform<float>(const int64_t elem_cnt,
                                                                const float min, const float max,
                                                                float* dptr);
template void RandomGenerator<DeviceType::kCPU>::Uniform<double>(const int64_t elem_cnt,
                                                                 const double min, const double max,
                                                                 double* dptr);

}  // namespace oneflow
