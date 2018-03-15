#ifndef ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class RandomGenerator;

template<DeviceType device_type, typename T>
struct RandomGeneratorUtil final {
  static void Uniform(RandomGenerator* rand_gen, const int64_t elem_cnt,
                      T* dptr);

  static void Uniform(RandomGenerator* rand_gen, const int64_t elem_cnt,
                      const T min, const T max, T* dptr);
};

class RandomGenerator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomGenerator);
  RandomGenerator(int64_t seed);
  ~RandomGenerator();

  template<DeviceType device_type, typename T>
  void Uniform(const int64_t elem_cnt, T* dptr) {
    RandomGeneratorUtil<device_type, T>::Uniform(this, elem_cnt, dptr);
  }

  //  Getters
  std::mt19937* mut_mt19937_generator() { return &mt19937_generator_; }

#ifdef WITH_CUDA
  curandGenerator_t* mut_curand_generator() { return &curand_generator_; }
#endif

 private:
  std::mt19937 mt19937_generator_;
#ifdef WITH_CUDA
  curandGenerator_t curand_generator_;
#endif
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
