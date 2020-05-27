#include "oneflow/customized/kernels/random_mask_generator.h"

namespace oneflow {

void RandomMaskGenerator<DeviceType::kCPU>::Generate(DeviceCtx* device_ctx, const int64_t n,
                                                     const float rate, int8_t* mask) {
  CHECK_GE(n, 0);
  std::uniform_real_distribution<float> random_distribution(GetZeroVal<float>(),
                                                            GetOneVal<float>());
  for (int64_t i = 0; i < n; ++i) { mask[i] = random_distribution(mt19937_generator_) > rate; }
}

template class RandomMaskGenerator<DeviceType::kCPU>;

}  // namespace oneflow
