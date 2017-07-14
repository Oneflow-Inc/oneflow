#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void SoftmaxForwardMaxGpu(const int64_t n, const int64_t w,
                                     const FloatingPointType* out,
                                     FloatingPointType* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    FloatingPointType max_value = out[i * w];
    for (int64_t j = 0; j < w; ++j) {
      max_value = max(max_value, out[i * w + j]);
    }
    tmp[i] = max_value;
  }
}

template<typename FloatingPointType>
__global__ void SoftmaxForwardSumGpu(const int64_t n, const int64_t w,
                                     const FloatingPointType* out,
                                     FloatingPointType* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    FloatingPointType sum_value = 0;
    for (int64_t j = 0; j < w; ++j) { sum_value += out[i * w + j]; }
    tmp[i] = sum_value;
  }
}

template<typename FloatingPointType>
__global__ void SoftmaxSubGpu(const int64_t n, const int64_t w,
                              FloatingPointType* matrix,
                              const FloatingPointType* vector) {
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] -= vector[i / w]; }
}

template<typename FloatingPointType>
__global__ void SoftmaxBackwardDotGpu(const int64_t n, const int64_t w,
                                      const FloatingPointType* out,
                                      const FloatingPointType* out_diff,
                                      FloatingPointType* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    FloatingPointType dot_result = 0;
    for (int64_t j = 0; j < w; ++j) {
      dot_result += out[i * w + j] * out_diff[i * w + j];
    }
    tmp[i] = dot_result;
  }
}

}  // namespace

template<typename FloatingPointType>
class SoftmaxKernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernelUtil);
  SoftmaxKernelUtil() = delete;

  static void ForwardMax(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp) {
    SoftmaxForwardMaxGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, w, out, tmp);
  }

  static void ForwardSum(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp) {
    SoftmaxForwardSumGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, w, out, tmp);
  }

  static void Sub(const KernelCtx& ctx, const int64_t n, const int64_t w,
                  FloatingPointType* matrix, const FloatingPointType* vector) {
    SoftmaxSubGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, w, matrix, vector);
  }

  static void BackwardDot(const KernelCtx& ctx, const int64_t n,
                          const int64_t w, const FloatingPointType* out,
                          const FloatingPointType* out_diff,
                          FloatingPointType* tmp) {
    SoftmaxBackwardDotGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, w, out, out_diff, tmp);
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(SoftmaxKernelUtil);

}  // namespace oneflow
