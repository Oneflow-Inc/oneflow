#include "oneflow/core/kernel/gelu_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void GeluForwardGpu(const int64_t n, const T* x, T* y) {
  UNIMPLEMENTED();
}

template<typename T>
__global__ void GeluBackwardGpu(const int64_t n, const T* x, const T* dy, T* dx) {
  UNIMPLEMENTED();
}

template<>
__global__ void GeluForwardGpu(const int64_t n, const float* x, float* y) {
  float inv_sqrt2 = sqrtf(0.5);
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 0.5 * x[i] * (1.0 + erff(inv_sqrt2 * x[i])); }
}

template<>
__global__ void GeluBackwardGpu(const int64_t n, const float* x, const float* dy, float* dx) {
  float inv_sqrt2 = sqrtf(0.5);
  float coef = sqrtf(2.0 / acosf(-1.0));
  CUDA_1D_KERNEL_LOOP(i, n) {
    dx[i] = 0.5 * (1.0 + erff(inv_sqrt2 * x[i]) + x[i] * coef * expf(-x[i] * x[i]));
  }
}

template<>
__global__ void GeluForwardGpu(const int64_t n, const double* x, double* y) {
  double inv_sqrt2 = sqrt(0.5);
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 0.5 * x[i] * (1.0 + erf(inv_sqrt2 * x[i])); }
}

template<>
__global__ void GeluBackwardGpu(const int64_t n, const double* x, const double* dy, double* dx) {
  double inv_sqrt2 = sqrt(0.5);
  double coef = sqrt(2.0 / acos(-1.0));
  CUDA_1D_KERNEL_LOOP(i, n) {
    dx[i] = 0.5 * (1.0 + erf(inv_sqrt2 * x[i]) + x[i] * coef * exp(-x[i] * x[i]));
  }
}

}  // namespace

template<typename T>
struct GeluKernelUtil<DeviceType::kGPU, T> {
  static void GeluForward(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    GeluForwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
  }

  static void GeluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* dy, T* dx) {
    GeluBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, dy, dx);
  }
};

#define INSTANTIATE_GELU_KERNEL_UTIL(type_cpp, type_proto) \
  template struct GeluKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GELU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
