#include "oneflow/core/kernel/sigmoid_cross_entropy_loss_kernel.h"

namespace oneflow {

namespace {
__global__ void ElementwiseMaxKernel(const int n, float* data, const float a) {
  CUDA_1D_KERNEL_LOOP(index, n) { data[index] = (data[index] > a) ? data[index] : a; }
}

template<typename PredType, typename LabelType>
__global__ void SigmoidCrossEntropyLossForward(const int64_t n, const PredType* prediction,
                                               const LabelType* label, PredType* loss,
                                               PredType* count) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (label[index] == -1) {
      loss[index] = 0.;
      count[index] = 0.;
    } else {
      loss[index] =
          -1. * prediction[index] * (label[index] - (prediction[index] >= 0))
          + logf(1 + expf(prediction[index] - 2 * prediction[index] * (prediction[index] >= 0)));
      count[index] = 1.;
    }
  }
}

template<typename PredType, typename LabelType>
__global__ void SigmoidCrossEntropyLossBackward(const int64_t n, const PredType* prediction,
                                                const LabelType* label, PredType* pred_diff,
                                                PredType* count) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (label[index] == -1) {
      pred_diff[index] = 0.;
      count[index] = 0.;
    } else {
      pred_diff[index] = 1. / (1. + expf(-prediction[index])) - label[index];
      count[index] = 1.;
    }
  }
}
}  // namespace

template<typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t n, const PredType* prediction,
                      const LabelType* label, PredType* loss, PredType* count) {
    SigmoidCrossEntropyLossForward<PredType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, prediction, label, loss, count);
  }

  static void Backward(DeviceCtx* ctx, const int64_t n, const PredType* prediction,
                       const LabelType* label, PredType* pred_diff, PredType* count) {
    SigmoidCrossEntropyLossBackward<PredType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, prediction, label, pred_diff, count);
  }
};

#define INSTANTIATE_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_UTIL(data_type_pair, label_type_pair) \
  template struct SigmoidCrossEntropyLossKernelUtil<                                        \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_UTIL,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}  // namespace oneflow
