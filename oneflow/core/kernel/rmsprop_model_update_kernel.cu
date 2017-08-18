#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void UpdateMeanSquareGpu(const int64_t n,
                                    const FloatingPointType alpha,
                                    const FloatingPointType decay_rate,
                                    FloatingPointType* mean_square,
                                    const FloatingPointType* model_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    mean_square[i] =
        alpha * model_diff[i] * model_diff[i] + decay_rate * mean_square[i];
  }
}

template<typename FloatingPointType>
__global__ void UpdateModelGpu(const int64_t n, FloatingPointType* model,
                               const FloatingPointType* model_diff,
                               const FloatingPointType* mean_square,
                               const FloatingPointType epsilon,
                               const FloatingPointType alpha) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    model[i] -= alpha * model_diff[i] / std::sqrt(mean_square[i] + epsilon);
  }
}

}  // namespace

template<typename FloatingPointType>
class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropMdUpdateKernelUtil);
  RMSPropMdUpdateKernelUtil() = delete;

  static void UpdateMeanSquare(const KernelCtx& ctx, const int64_t n,
                               const FloatingPointType alpha,
                               const FloatingPointType decay_rate,
                               FloatingPointType* mean_square,
                               const FloatingPointType* model_diff) {
    UpdateMeanSquareGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, alpha, decay_rate, mean_square,
                                            model_diff);
  }

  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          FloatingPointType* model,
                          const FloatingPointType* model_diff,
                          const FloatingPointType* mean_square,
                          const FloatingPointType epsilon,
                          const FloatingPointType alpha) {
    UpdateModelGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, model, model_diff, mean_square,
                                            epsilon, alpha);
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(RMSPropMdUpdateKernelUtil);

}  // namespace oneflow
