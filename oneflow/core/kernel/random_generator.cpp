#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

RandomGenerator::RandomGenerator(int64_t seed) : mt19937_generator_(seed) {
#ifdef WITH_CUDA
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS) {
    CudaCheck(curandSetPseudoRandomGeneratorSeed(curand_generator_, GetCurTime()));
  }
#endif
}

RandomGenerator::~RandomGenerator() {
#ifdef WITH_CUDA
  CudaCheck(curandDestroyGenerator(curand_generator_));
#endif
}

template<typename T>
struct RandomGeneratorUtil<DeviceType::kCPU, T> final {
  static void Uniform(RandomGenerator* rand_gen, const int64_t elem_cnt, T* dptr) {
    Uniform(rand_gen, elem_cnt, ZeroVal<T>::value, OneVal<T>::value, dptr);
  }

  static void Uniform(RandomGenerator* rand_gen, const int64_t elem_cnt, const T min, const T max,
                      T* dptr) {
    CHECK_GE(elem_cnt, 0);
    CHECK(dptr);
    CHECK_LE(min, max);
    std::uniform_real_distribution<T> random_distribution(
        min, std::nextafter(max, std::numeric_limits<T>::max()));

    for (int64_t i = 0; i < elem_cnt; ++i) {
      dptr[i] = random_distribution(*rand_gen->mut_mt19937_generator());
    }
  }
};

#define INITIATE_RANDOM_GENERATOR_UTIL(T, type_proto) \
  template struct RandomGeneratorUtil<DeviceType::kCPU, T>;

OF_PP_FOR_EACH_TUPLE(INITIATE_RANDOM_GENERATOR_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
