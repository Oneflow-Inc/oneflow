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

RandomGeneratorGpuImpl::RandomGeneratorGpuImpl(int64_t seed, cudaStream_t cuda_stream) {
  CudaCheck(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CudaCheck(curandSetPseudoRandomGeneratorSeed(curand_generator_, seed));
  CudaCheck(curandSetStream(curand_generator_, cuda_stream));
}

RandomGeneratorGpuImpl::~RandomGeneratorGpuImpl() {
  CudaCheck(curandDestroyGenerator(curand_generator_));
}

template<typename T>
void RandomGeneratorGpuImpl::TUniform(const int64_t elem_cnt, T* dptr) {
  RngUniformGpu(curand_generator_, elem_cnt, dptr);
}

template void RandomGeneratorGpuImpl::TUniform<float>(const int64_t elem_cnt, float* dptr);
template void RandomGeneratorGpuImpl::TUniform<double>(const int64_t elem_cnt, double* dptr);

}  // namespace oneflow
