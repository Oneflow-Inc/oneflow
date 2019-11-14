#include "oneflow/core/kernel/util/cuda_dnn_interface.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

#define HALF_CHECK_FAILED                                             \
  printf("half operations are only supported when CUDA_ARCH >= 530"); \
  assert(false)

__inline__ __device__ half hone() { return __float2half(1.0); }
__inline__ __device__ half hzero() { return __float2half(0.0); }

template<typename T>
__global__ void ReluForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : 0; }
}

template<>
__global__ void ReluForwardGpu<half>(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (__hgt(x[i], hzero())) {
      y[i] = x[i];
    } else {
      y[i] = hzero();
    }
  }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

template<typename T>
__global__ void InplaceReluForwardGpu(const int n, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    // There is a subtle cuda bug in (y[i] <= 0)
    if (!(y[i] > 0)) { y[i] = 0; }
  }
}

template<>
__global__ void InplaceReluForwardGpu<half>(const int n, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (!__hgt(y[i], hzero())) { y[i] = hzero(); }
  }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

template<typename T>
__global__ void ReluBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = y[i] > 0 ? dy[i] : 0; }
}

template<>
__global__ void ReluBackwardGpu<half>(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half zero = __float2half(0.0);
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (__hgt(y[i], zero)) {
      dx[i] = dy[i];
    } else {
      dx[i] = zero;
    }
  }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

template<typename T>
__global__ void InplaceReluBackwardGpu(const int n, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (!(y[i] > 0)) { dx[i] = 0; }
  }
}

template<>
__global__ void InplaceReluBackwardGpu<half>(const int n, const half* y, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half zero = __float2half(0.0);
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (!__hgt(y[i], zero)) { dx[i] = zero; }
  }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

template<typename T>
__global__ void SigmoidForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 1.0 / (1.0 + std::exp(-x[i])); }
}

template<>
__global__ void SigmoidForwardGpu<half>(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = __hdiv(hone(), __hadd(hone(), hexp(__hneg(x[i])))); }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

template<typename T>
__global__ void SigmoidBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dy[i] * y[i] * (1.0 - y[i]); }
}

template<>
__global__ void SigmoidBackwardGpu<half>(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = __hmul(dy[i], __hmul(y[i], __hsub(hone(), y[i]))); }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

template<typename T>
__global__ void TanHForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::tanh(x[i]); }
}

template<>
__global__ void TanHForwardGpu<half>(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    half ex = hexp(x[i]);
    half e_x = hexp(__hneg(x[i]));
    y[i] = __hdiv(__hsub(ex, e_x), __hadd(ex, e_x));
  }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

template<typename T>
__global__ void TanHBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dy[i] * (1.0 - y[i] * y[i]); }
}

template<>
__global__ void TanHBackwardGpu<half>(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = __hmul(dy[i], __hsub(hone(), __hmul(y[i], y[i]))); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

template<typename T>
struct ReluHelper final {
  static void ReluForward(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    if (x == y) {
      InplaceReluForwardGpu<T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y);
    } else {
      ReluForwardGpu<T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
    }
  }

  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* y, const T* dy, T* dx) {
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    if (dy == dx) {
      InplaceReluBackwardGpu<T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dx);
    } else {
      ReluBackwardGpu<T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy,
                                                                                        dx);
    }
  }
};

}  // namespace

void DnnIf<DeviceType::kGPU>::Relu(DeviceCtx* ctx, const int64_t n, const float* x, float* y) {
  ReluHelper<float>::ReluForward(ctx, n, x, y);
}

void DnnIf<DeviceType::kGPU>::Relu(DeviceCtx* ctx, const int64_t n, const double* x, double* y) {
  ReluHelper<double>::ReluForward(ctx, n, x, y);
}

void DnnIf<DeviceType::kGPU>::Relu(DeviceCtx* ctx, const int64_t n, const float16* x, float16* y) {
  ReluHelper<half>::ReluForward(ctx, n, reinterpret_cast<const half*>(x),
                                reinterpret_cast<half*>(y));
}

void DnnIf<DeviceType::kGPU>::ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x,
                                           const float* y, const float* dy, float* dx) {
  ReluHelper<float>::ReluBackward(ctx, n, y, dy, dx);
}

void DnnIf<DeviceType::kGPU>::ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x,
                                           const double* y, const double* dy, double* dx) {
  ReluHelper<double>::ReluBackward(ctx, n, y, dy, dx);
}

void DnnIf<DeviceType::kGPU>::ReluBackward(DeviceCtx* ctx, const int64_t n, const float16* x,
                                           const float16* y, const float16* dy, float16* dx) {
  ReluHelper<half>::ReluBackward(ctx, n, reinterpret_cast<const half*>(y),
                                 reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
}

void DnnIf<DeviceType::kGPU>::Sigmoid(DeviceCtx* ctx, int64_t n, const float* x, float* y) {
  CHECK(IsKernelSafeInt32(n));
  SigmoidForwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

void DnnIf<DeviceType::kGPU>::Sigmoid(DeviceCtx* ctx, int64_t n, const double* x, double* y) {
  CHECK(IsKernelSafeInt32(n));
  SigmoidForwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

void DnnIf<DeviceType::kGPU>::Sigmoid(DeviceCtx* ctx, int64_t n, const float16* x, float16* y) {
  CHECK(IsKernelSafeInt32(n));
  SigmoidForwardGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
}

void DnnIf<DeviceType::kGPU>::SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float* x,
                                              const float* y, const float* dy, float* dx) {
  CHECK(IsKernelSafeInt32(n));
  SigmoidBackwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

void DnnIf<DeviceType::kGPU>::SigmoidBackward(DeviceCtx* ctx, const int64_t n, const double* x,
                                              const double* y, const double* dy, double* dx) {
  CHECK(IsKernelSafeInt32(n));
  SigmoidBackwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

void DnnIf<DeviceType::kGPU>::SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float16* x,
                                              const float16* y, const float16* dy, float16* dx) {
  CHECK(IsKernelSafeInt32(n));
  SigmoidBackwardGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy),
          reinterpret_cast<half*>(dx));
}

void DnnIf<DeviceType::kGPU>::TanH(DeviceCtx* ctx, int64_t n, const float* x, float* y) {
  CHECK(IsKernelSafeInt32(n));
  TanHForwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

void DnnIf<DeviceType::kGPU>::TanH(DeviceCtx* ctx, int64_t n, const double* x, double* y) {
  CHECK(IsKernelSafeInt32(n));
  TanHForwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

void DnnIf<DeviceType::kGPU>::TanH(DeviceCtx* ctx, int64_t n, const float16* x, float16* y) {
  CHECK(IsKernelSafeInt32(n));
  TanHForwardGpu<half><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
}

void DnnIf<DeviceType::kGPU>::TanHBackward(DeviceCtx* ctx, const int64_t n, const float* x,
                                           const float* y, const float* dy, float* dx) {
  CHECK(IsKernelSafeInt32(n));
  TanHBackwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

void DnnIf<DeviceType::kGPU>::TanHBackward(DeviceCtx* ctx, const int64_t n, const double* x,
                                           const double* y, const double* dy, double* dx) {
  CHECK(IsKernelSafeInt32(n));
  TanHBackwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

void DnnIf<DeviceType::kGPU>::TanHBackward(DeviceCtx* ctx, const int64_t n, const float16* x,
                                           const float16* y, const float16* dy, float16* dx) {
  CHECK(IsKernelSafeInt32(n));
  TanHBackwardGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy),
          reinterpret_cast<half*>(dx));
}

}  // namespace oneflow
