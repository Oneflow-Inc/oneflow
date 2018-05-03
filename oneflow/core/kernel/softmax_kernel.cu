#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SoftmaxForwardMaxGpu(const int64_t n, const int64_t w, const T* out, T* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T max_value = out[i * w];
    for (int64_t j = 0; j < w; ++j) {
      max_value = max_value > out[i * w + j] ? max_value : out[i * w + j];
    }
    tmp[i] = max_value;
  }
}

template<typename T>
__global__ void SoftmaxForwardSumGpu(const int64_t n, const int64_t w, const T* out, T* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T sum_value = 0;
    for (int64_t j = 0; j < w; ++j) { sum_value += out[i * w + j]; }
    tmp[i] = sum_value;
  }
}

template<typename T>
__global__ void SoftmaxSubGpu(const int64_t n, const int64_t w, T* matrix, const T* vector) {
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] -= vector[i / w]; }
}

template<typename T>
__global__ void SoftmaxBackwardDotGpu(const int64_t n, const int64_t w, const T* out,
                                      const T* out_diff, T* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T dot_result = 0;
    for (int64_t j = 0; j < w; ++j) { dot_result += out[i * w + j] * out_diff[i * w + j]; }
    tmp[i] = dot_result;
  }
}

}  // namespace

template<typename T>
struct SoftmaxKernelUtil<DeviceType::kGPU, T> {
  static void ForwardMax(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* out, T* tmp) {
    SoftmaxForwardMaxGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, w, out,
                                                                                      tmp);
  }

  static void ForwardSum(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* out, T* tmp) {
    SoftmaxForwardSumGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, w, out,
                                                                                      tmp);
  }

  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector) {
    SoftmaxSubGpu<T>
        <<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, w, matrix, vector);
  }

  static void BackwardDot(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* out,
                          const T* out_diff, T* tmp) {
    SoftmaxBackwardDotGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, w, out, out_diff, tmp);
  }
};

#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
