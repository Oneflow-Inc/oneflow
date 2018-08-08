#include "oneflow/core/kernel/sigmoid_cross_entropy_loss_kernel.h"

namespace oneflow {

namespace {
template<typename PredType>
__global__ void NoSmallerThan(const int n, PredType* x, const float floor_val) {
  CUDA_1D_KERNEL_LOOP(index, n) { x[index] = (x[index] > floor_val) ? x[index] : floor_val; }
}

template<typename PredType, typename LabelType>
__global__ void SigmoidCrossEntropyLossForward(const int64_t n, const PredType* prediction,
                                               const LabelType* label, PredType* loss,
                                               PredType* count) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (label[index] == -1) {
      loss[index] = 0.f;
      count[index] = 0.f;
    } else {
      loss[index] =
          -1.f * prediction[index] * (label[index] - (prediction[index] >= 0))
          + logf(1 + expf(prediction[index] - 2 * prediction[index] * (prediction[index] >= 0)));
      count[index] = 1.f;
    }
  }
}

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
struct SigmoidCrossEntropyLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                      const PredType* prediction, const LabelType* label, PredType* loss,
                      PredType* count, PredType* normalize, PredType* average_loss) {
    SigmoidCrossEntropyLossForward<PredType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, prediction, label, loss, count);
    KernelUtil<DeviceType::kGPU, PredType>::Sum(
        ctx, n, loss, average_loss, static_cast<PredType*>(ctx->buf_ptr()), ctx->buf_size());
    if (conf.normalize()) {
      KernelUtil<DeviceType::kGPU, PredType>::Sum(
          ctx, n, count, normalize, static_cast<PredType*>(ctx->buf_ptr()), ctx->buf_size());
      NoSmallerThan<PredType>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              n, normalize, 1e-5);
      KernelUtil<DeviceType::kGPU, PredType>::Div(ctx, 1, average_loss, normalize);
    }
    KernelUtil<DeviceType::kGPU, PredType>::Scal(ctx, 1, average_loss, average_loss, conf.scale());
  }

  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                       const PredType* prediction, const LabelType* label, PredType* pred_diff,
                       const PredType* normalize) {
    SigmoidCrossEntropyLossBackward<PredType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, prediction, label, pred_diff);
    KernelUtil<DeviceType::kGPU, PredType>::Scal(ctx, n, pred_diff, pred_diff, conf.scale());
    if (conf.normalize()) {
      KernelUtil<DeviceType::kGPU, PredType>::Scal(ctx, n, pred_diff, pred_diff, 1 / *normalize);
    }
  }
};

#define INSTANTIATE_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_UTIL(data_type_pair, label_type_pair) \
  template struct SigmoidCrossEntropyLossKernelUtil<                                        \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_UTIL,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}  // namespace oneflow
