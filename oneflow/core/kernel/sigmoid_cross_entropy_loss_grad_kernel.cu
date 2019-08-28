#include "oneflow/core/kernel/sigmoid_cross_entropy_loss_grad_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void SigmoidCrossEntropyLossBackward(const int64_t n, const PredType* prediction,
                                                const LabelType* label, PredType* pred_diff) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (label[index] == -1) {
      pred_diff[index] = 0.f;
    } else {
      pred_diff[index] = 1.f / (1.f + expf(-prediction[index])) - label[index];
    }
  }
}

}  // namespace

template <typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossGradKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  
  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossGradOpConf& conf, const int64_t n,
                       const PredType* prediction, const LabelType* label, PredType* pred_diff) {
    SigmoidCrossEntropyLossBackward<PredType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, prediction, label, pred_diff);
    }
};

// instantiate template declaration
template struct SigmoidCrossEntropyLossGradKernelUtil<DeviceType::kGPU, float, float>;

}  // namespace oneflow
