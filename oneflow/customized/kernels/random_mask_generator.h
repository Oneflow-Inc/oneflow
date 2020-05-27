#ifndef ONEFLOW_CUSTOMIZED_KERNELS_RANDOM_MASK_GENERATOR_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_RANDOM_MASK_GENERATOR_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/device_context.h"
#include <curand.h>
#include <curand_kernel.h>

namespace oneflow {

template<DeviceType device_type>
class RandomMaskGenerator;

template<>
class RandomMaskGenerator<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomMaskGenerator);
  RandomMaskGenerator(int64_t seed) : mt19937_generator_(seed) {}
  ~RandomMaskGenerator() {}

  void Generate(DeviceCtx* device_ctx, int64_t n, float rate, int8_t* mask);

 private:
  std::mt19937 mt19937_generator_;
};

template<>
class RandomMaskGenerator<DeviceType::kGPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomMaskGenerator);
  RandomMaskGenerator(int64_t seed);
  ~RandomMaskGenerator();

  void Generate(DeviceCtx* device_ctx, int64_t n, float rate, int8_t* mask);

 private:
  curandState* curand_states_;
  int32_t block_num_;
  int32_t thread_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_RANDOM_MASK_GENERATOR_H_
