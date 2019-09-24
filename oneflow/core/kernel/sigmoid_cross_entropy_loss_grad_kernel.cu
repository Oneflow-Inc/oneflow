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

template<typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossGradKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossGradOpConf& conf,
                       const int64_t n, const PredType* prediction, const LabelType* label,
                       PredType* pred_diff) {
    SigmoidCrossEntropyLossBackward<PredType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, prediction, label, pred_diff);
  }
};

#define INSTANTIATE_SIGMOID_CROSS_ENTROPY_LOSS_GRAD_KERNEL_UTIL(data_type_pair, label_type_pair) \
  template struct SigmoidCrossEntropyLossGradKernelUtil<                                         \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SIGMOID_CROSS_ENTROPY_LOSS_GRAD_KERNEL_UTIL,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
