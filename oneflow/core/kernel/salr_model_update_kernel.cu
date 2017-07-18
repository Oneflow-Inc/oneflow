#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/salr_model_update_kernel.h"

namespace oneflow {

namespace {

// if diff(t) * diff(t-1) > 0
// then learning_rate = learning_rate + delta
// else learning_rate = learning_rate * (1 - delta)
template<typename FloatingPointType>
__global__ void UpdateLearningRateGpu(const int64_t n,
                                      const FloatingPointType delta,
                                      FloatingPointType* last_diff_flag,
                                      const FloatingPointType* model_diff,
                                      FloatingPointType* learning_rate) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (last_diff_flag[i] * model_diff[i] > 0) {
      learning_rate[i] = learning_rate[i] + delta;
    } else {
      learning_rate[i] = learning_rate[i] * (1 - delta);
    }
    last_diff_flag[i] = model_diff[i] > 0 ? 1 : -1;
  }
}

// model -= (-epsilon) * learning_rate * model_diff
template<typename FloatingPointType>
__global__ void UpdateModelGpu(const int64_t n, FloatingPointType* model,
                               const FloatingPointType* model_diff,
                               const FloatingPointType* learning_rate,
                               const FloatingPointType epsilon) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    model[i] -= -epsilon * learning_rate[i] * model_diff[i];
  }
}

}  // namespace

template<typename FloatingPointType>
class SALRMdUpdateKernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SALRMdUpdateKernelUtil);
  SALRMdUpdateKernelUtil() = delete;

  static void UpdateLearningRate(const KernelCtx& ctx, const int64_t n,
                                 const FloatingPointType delta,
                                 FloatingPointType* last_diff_flag,
                                 const FloatingPointType* model_diff,
                                 FloatingPointType* learning_rate) {
    UpdateLearningRateGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, delta, last_diff_flag,
                                            model_diff, learning_rate);
  }

  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          FloatingPointType* model,
                          const FloatingPointType* model_diff,
                          const FloatingPointType* learning_rate,
                          const FloatingPointType epsilon) {
    UpdateModelGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, model, model_diff, learning_rate,
                                            epsilon);
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(SALRMdUpdateKernelUtil);

}  // namespace oneflow
