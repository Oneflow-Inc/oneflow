#include "oneflow/core/kernel/gelu_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void GeluForwardGpu(const int64_t n, const T* x, const T inv_sqrt2, T* y) {
  UNIMPLEMENTED();
}

template<typename T>
__global__ void GeluBackwardGpu(const int64_t n, const T* x, const T* dy, const T inv_sqrt2,
                                const T coef, T* dx) {
  UNIMPLEMENTED();
}

template<>
__global__ void GeluForwardGpu(const int64_t n, const float* x, const float inv_sqrt2, float* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 0.5f * x[i] * (1.0f + erff(inv_sqrt2 * x[i])); }
}

template<>
__global__ void GeluBackwardGpu(const int64_t n, const float* x, const float* dy,
                                const float inv_sqrt2, const float coef, float* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dx[i] =
        0.5f * (1.0f + erff(inv_sqrt2 * x[i]) + x[i] * coef * expf(-0.5f * x[i] * x[i])) * dy[i];
  }
}

template<>
__global__ void GeluForwardGpu(const int64_t n, const double* x, const double inv_sqrt2,
                               double* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 0.5 * x[i] * (1.0 + erf(inv_sqrt2 * x[i])); }
}

template<>
__global__ void GeluBackwardGpu(const int64_t n, const double* x, const double* dy,
                                const double inv_sqrt2, const double coef, double* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dx[i] = 0.5 * (1.0 + erf(inv_sqrt2 * x[i]) + x[i] * coef * exp(-0.5 * x[i] * x[i])) * dy[i];
  }
}

__global__ void GeluForwardGpuHalf(const int64_t n, const half* x, const float inv_sqrt2, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    float f_x = __half2float(x[i]);
    y[i] = __float2half(0.5f * f_x * (1.0f + erff(inv_sqrt2 * f_x)));
  }
}

__global__ void GeluBackwardGpuHalf(const int64_t n, const half* x, const half* dy,
                                    const float inv_sqrt2, const float coef, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    float f_x = __half2float(x[i]);
    dx[i] =
        __float2half(0.5f * (1.0f + erff(inv_sqrt2 * f_x) + f_x * coef * expf(-0.5f * f_x * f_x))
                     * __half2float(dy[i]));
  }
}

}  // namespace

template<typename T>
struct GeluKernelUtil<DeviceType::kGPU, T> {
  static void GeluForward(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    const T inv_sqrt2 = sqrt(0.5);
    GeluForwardGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, x, inv_sqrt2, y);
  }

  static void GeluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* dy, T* dx) {
    const T inv_sqrt2 = sqrt(0.5);
    const T coef = sqrt(2.0 / acos(-1.0));
    GeluBackwardGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, x, dy, inv_sqrt2, coef, dx);
  }
};

template<>
struct GeluKernelUtil<DeviceType::kGPU, float16> {
  static void GeluForward(DeviceCtx* ctx, const int64_t n, const float16* x, float16* y) {
    const float inv_sqrt2 = sqrt(0.5);
    GeluForwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(x), inv_sqrt2, reinterpret_cast<half*>(y));
  }

  static void GeluBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* dy,
                           float16* dx) {
    const float inv_sqrt2 = sqrt(0.5);
    const float coef = sqrt(2.0 / acos(-1.0));
    GeluBackwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                          ctx->cuda_stream()>>>(n, reinterpret_cast<const half*>(x),
                                                reinterpret_cast<const half*>(dy), inv_sqrt2, coef,
                                                reinterpret_cast<half*>(dx));
  }
};

#define INSTANTIATE_GELU_KERNEL_UTIL(type_cpp, type_proto) \
  template struct GeluKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GELU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
