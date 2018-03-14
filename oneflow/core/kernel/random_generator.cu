#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

namespace {

template<typename T>
void RngUniformGpu(const curandGenerator_t& gen, int64_t n, T* ret);

template<>
void RngUniformGpu<float>(const curandGenerator_t& gen, int64_t n, float* ret) {
  CudaCheck(curandGenerateUniform(gen, ret, n));
}

template<>
void RngUniformGpu<double>(const curandGenerator_t& gen, int64_t n,
                           double* ret) {
  CudaCheck(curandGenerateUniformDouble(gen, ret, n));
}

}  // namespace

template<typename T>
struct RandomGeneratorUtil<DeviceType::kGPU, T> final {
  static void Uniform(RandomGenerator* rand_gen, const int64_t elem_cnt,
                      T* dptr) {
    RngUniformGpu(*rand_gen->mut_curand_generator(), elem_cnt, dptr);
  }

  static void Uniform(RandomGenerator* rand_gen, const int64_t elem_cnt,
                      const T min, const T max, T* dptr) {
    UNIMPLEMENTED();
  }
};

#define INITIATE_RANDOM_GENERATOR_UTIL(T, type_proto) \
  template struct RandomGeneratorUtil<DeviceType::kGPU, T>;

OF_PP_FOR_EACH_TUPLE(INITIATE_RANDOM_GENERATOR_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
