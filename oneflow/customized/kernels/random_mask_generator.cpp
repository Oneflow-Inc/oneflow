#include "oneflow/customized/kernels/random_mask_generator.h"

namespace oneflow {

void RandomMaskGenerator<DeviceType::kCPU>::Generate(const int64_t elem_cnt, const float rate,
                                                     int8_t* mask) {
  CHECK_GE(elem_cnt, 0);
  std::uniform_real_distribution<float> random_distribution(
      GetZeroVal<float>(), std::nextafter(GetOneVal<float>(), GetMaxVal<float>()));
  for (int64_t i = 0; i < elem_cnt; ++i) {
    mask[i] = random_distribution(mt19937_generator_) > rate;
  }
}

template class RandomMaskGenerator<DeviceType::kCPU>;

}  // namespace oneflow
