#ifndef ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class RandomGenerator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomGenerator);
  RandomGenerator() = default;
  virtual ~RandomGenerator() {}

  template<typename T>
  void Uniform(const int64_t elem_cnt, T* dptr) {
    VUniform(elem_cnt, dptr);
  }

 private:
  virtual void VUniform(const int64_t elem_cnt, float* dptr) = 0;
  virtual void VUniform(const int64_t elem_cnt, double* dptr) = 0;
};

template<typename Impl>
class RandomGeneratorIf : public Impl {
 public:
  template<class... TArgs>
  RandomGeneratorIf(TArgs&&... args) : Impl(std::forward<TArgs>(args)...) {}

 private:
  void VUniform(const int64_t elem_cnt, float* dptr) final { Impl::TUniform(elem_cnt, dptr); }
  void VUniform(const int64_t elem_cnt, double* dptr) final { Impl::TUniform(elem_cnt, dptr); }
};

class RandomGeneratorCpuImpl : public RandomGenerator {
 public:
  RandomGeneratorCpuImpl(int64_t seed) : mt19937_generator_(seed) {}

  template<typename T>
  void TUniform(const int64_t elem_cnt, T* dptr);
  template<typename T>
  void TUniform(const int64_t elem_cnt, const T min, const T max, T* dptr);

 private:
  std::mt19937 mt19937_generator_;
};

using RandomGeneratorCpu = RandomGeneratorIf<RandomGeneratorCpuImpl>;

class RandomGeneratorGpuImpl : public RandomGenerator {
 public:
  RandomGeneratorGpuImpl(int64_t seed, cudaStream_t cuda_stream);
  ~RandomGeneratorGpuImpl();

  template<typename T>
  void TUniform(const int64_t elem_cnt, T* dptr);

 private:
  curandGenerator_t curand_generator_;
};

using RandomGeneratorGpu = RandomGeneratorIf<RandomGeneratorGpuImpl>;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
