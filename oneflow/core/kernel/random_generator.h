#ifndef ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

template<DeviceType device_type>
class RandomGenerator;

template<>
class RandomGenerator<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomGenerator);
  RandomGenerator(int64_t seed, DeviceCtx* device_ctx) : mt19937_generator_(seed) {}
  ~RandomGenerator() {}

  template<typename T>
  void Uniform(const int64_t elem_cnt, T* dptr);
  template<typename T>
  void Uniform(const int64_t elem_cnt, const T min, const T max, T* dptr);

 private:
  std::mt19937 mt19937_generator_;
};

template<>
class RandomGenerator<DeviceType::kGPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomGenerator);
  RandomGenerator(int64_t seed, DeviceCtx* device_ctx);
  ~RandomGenerator();

  template<typename T>
  void Uniform(const int64_t elem_cnt, T* dptr);

 private:
  curandGenerator_t curand_generator_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
