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
void RngUniformGpu<double>(const curandGenerator_t& gen, int64_t n, double* ret) {
  CudaCheck(curandGenerateUniformDouble(gen, ret, n));
}

}  // namespace

RandomGenerator<DeviceType::kGPU>::RandomGenerator(int64_t seed, DeviceCtx* device_ctx) {
  CHECK_NOTNULL(device_ctx);
  CudaCheck(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CudaCheck(curandSetPseudoRandomGeneratorSeed(curand_generator_, seed));
  CudaCheck(curandSetStream(curand_generator_, device_ctx->cuda_stream()));
}

RandomGenerator<DeviceType::kGPU>::~RandomGenerator() {
  CudaCheck(curandDestroyGenerator(curand_generator_));
}

template<typename T>
void RandomGenerator<DeviceType::kGPU>::Uniform(const int64_t elem_cnt, T* dptr) {
  RngUniformGpu(curand_generator_, elem_cnt, dptr);
}

#define INITIATE_GPU_RANDOM_GENERATOR_UNIFORM(T, typeproto) \
  template void RandomGenerator<DeviceType::kGPU>::Uniform<T>(const int64_t elem_cnt, T* dptr);

OF_PP_FOR_EACH_TUPLE(INITIATE_GPU_RANDOM_GENERATOR_UNIFORM, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
