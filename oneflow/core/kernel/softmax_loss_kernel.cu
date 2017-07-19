#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/softmax_loss_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void SoftmaxLossForwardTmp(const int64_t n, const int64_t w,
                                      const FloatingPointType* label,
                                      const FloatingPointType* prob,
                                      FloatingPointType* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    tmp[i] = -std::log(prob[i * w + static_cast<int64_t>(label[i])]);
  }
}

template<typename FloatingPointType>
__global__ void SoftmaxLossBackwardSub(const int64_t n, const int64_t w,
                                       const FloatingPointType* label,
                                       FloatingPointType* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    in_diff[i * w + static_cast<int64_t>(label[i])] -= 1;
  }
}

}  // namespace

template<typename FloatingPointType>
class SoftmaxLossKernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxLossKernelUtil);
  SoftmaxLossKernelUtil() = delete;

  static void ComputeLoss(const KernelCtx& ctx, const int64_t n,
                          const int64_t w, const FloatingPointType* label,
                          const FloatingPointType* prob, FloatingPointType* tmp,
                          FloatingPointType* loss) {
    SoftmaxLossForwardTmp<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, w, label, prob, tmp);
    KernelUtil<DeviceType::kGPU, FloatingPointType>::Sum(ctx, n, tmp, loss);
    KernelUtil<DeviceType::kGPU, FloatingPointType>::BlasScal(ctx, 1, 1.0 / n,
                                                              loss, 1);
  }

  static void BackwardSub(const KernelCtx& ctx, const int64_t n,
                          const int64_t w, const FloatingPointType* label,
                          FloatingPointType* in_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        in_diff[i * w + static_cast<int64_t>(label[i])] -= 1;
      }
    });
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(SoftmaxLossKernelUtil);

}  // namespace oneflow
