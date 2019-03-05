#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void SoftmaxSubGpu(const int64_t n, const int64_t w, T* matrix, const T* vector) {
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] -= vector[i / w]; }
}

template<typename T>
__global__ void SoftmaxDivGpu(const int64_t n, const int64_t w, T* matrix, const T* vector) {
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] /= vector[i / w]; }
}

__global__ void SoftmaxSubHalfGpu(const int64_t n, const int64_t w, half* matrix,
                                  const half* vector) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] = __hsub(matrix[i], vector[i / w]); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void SoftmaxDivHalfGpu(const int64_t n, const int64_t w, half* matrix,
                                  const half* vector) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] = __hdiv(matrix[i], vector[i / w]); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

}  // namespace

template<typename T>
struct SoftmaxKernelUtil<DeviceType::kGPU, T> {
  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector) {
    SoftmaxSubGpu<T>
        <<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, w, matrix, vector);
  }

  static void Div(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector) {
    SoftmaxDivGpu<T>
        <<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, w, matrix, vector);
  }
};

template<>
struct SoftmaxKernelUtil<DeviceType::kGPU, float16> {
  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, float16* matrix,
                  const float16* vector) {
    SoftmaxSubHalfGpu<<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(n, w, reinterpret_cast<half*>(matrix),
                                              reinterpret_cast<const half*>(vector));
  }

  static void Div(DeviceCtx* ctx, const int64_t n, const int64_t w, float16* matrix,
                  const float16* vector) {
    SoftmaxDivHalfGpu<<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(n, w, reinterpret_cast<half*>(matrix),
                                              reinterpret_cast<const half*>(vector));
  }
};

#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
