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

RandomGenerator<DeviceType::kGPU>::RandomGenerator(int64_t seed, cudaStream_t cuda_stream) {
  CudaCheck(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CudaCheck(curandSetPseudoRandomGeneratorSeed(curand_generator_, seed));
  CudaCheck(curandSetStream(curand_generator_, cuda_stream));
}

RandomGenerator<DeviceType::kGPU>::~RandomGenerator() {
  CudaCheck(curandDestroyGenerator(curand_generator_));
}

template<typename T>
void RandomGenerator<DeviceType::kGPU>::Uniform(const int64_t elem_cnt, T* dptr) {
  RngUniformGpu(curand_generator_, elem_cnt, dptr);
}

template void RandomGenerator<DeviceType::kGPU>::Uniform<float>(const int64_t elem_cnt,
                                                                float* dptr);
template void RandomGenerator<DeviceType::kGPU>::Uniform<double>(const int64_t elem_cnt,
                                                                 double* dptr);

}  // namespace oneflow
