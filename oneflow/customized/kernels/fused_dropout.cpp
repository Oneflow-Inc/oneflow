#include "oneflow/customized/kernels/fused_dropout.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename T>
void FusedDropout<DeviceType::kCPU>::Dropout(const int64_t elem_cnt, const float rate,
                                             const float scale, const T* x, T* y, int8_t* mask) {
  CHECK_GE(elem_cnt, 0);
  std::uniform_real_distribution<float> random_distribution(
      GetZeroVal<float>(), std::nextafter(GetOneVal<float>(), GetMaxVal<float>()));
  for (int64_t i = 0; i < elem_cnt; ++i) {
    float random = random_distribution(mt19937_generator_);
    y[i] = x[i] * static_cast<T>(random > rate) * scale;
  }
}

#define INITIATE_CPU_FUSED_DROPOUT(T, typeproto)                                                \
  template void FusedDropout<DeviceType::kCPU>::Dropout<T>(const int64_t elem_cnt,              \
                                                           const float rate, const float scale, \
                                                           const T* x, T* y, int8_t* mask);

OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_FUSED_DROPOUT, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
