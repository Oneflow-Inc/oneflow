#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void MultinomialLogisticLossForwardGpu(
    const int64_t piece_size, const int64_t num_of_classes,
    const FloatingPointType* prediction, const FloatingPointType* labels,
    FloatingPointType* loss_buff) {
  CUDA_1D_KERNEL_LOOP(i, piece_size) {
    int64_t label = labels[i];
    FloatingPointType prob = prediction[i * num_of_classes + label];
    prob = prob > FloatingPointType(kLOG_THRESHOLD)
               ? prob
               : FloatingPointType(kLOG_THRESHOLD);
    loss_buff[i] = -logf(prob) / piece_size;
  }
}

template<typename FloatingPointType>
__global__ void MultinomialLogisticLossBackwardGpu(
    const int64_t piece_size, const int64_t num_of_classes,
    const FloatingPointType* prediction, const FloatingPointType* labels,
    FloatingPointType* prediction_diff) {
  const FloatingPointType scale = -1.0 / piece_size;
  CUDA_1D_KERNEL_LOOP(i, piece_size) {
    int64_t label = labels[i];
    FloatingPointType prob = prediction[i * num_of_classes + label];
    prob = prob > FloatingPointType(kLOG_THRESHOLD)
               ? prob
               : FloatingPointType(kLOG_THRESHOLD);
    prediction_diff[i * num_of_classes + label] = scale / prob;
  }
}

}  // namespace

template<typename FloatingPointType>
class MultinomialLogisticLossKernelUtil<DeviceType::kGPU, FloatingPointType>
    final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t piece_size,
                      const int64_t num_of_classes,
                      const FloatingPointType* prediction,
                      const FloatingPointType* labels, FloatingPointType* loss,
                      FloatingPointType* loss_buff) {
    MultinomialLogisticLossForwardGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(piece_size), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(piece_size, num_of_classes,
                                            prediction, labels, loss_buff);
    KernelUtil<DeviceType::kGPU, FloatingPointType>::Sum(
        ctx, piece_size, loss_buff, loss, loss_buff,
        sizeof(FloatingPointType) * piece_size);
  }

  static void Backward(const KernelCtx& ctx, const int64_t piece_size,
                       const int64_t num_of_classes,
                       const FloatingPointType* prediction,
                       const FloatingPointType* labels,
                       FloatingPointType* prediction_diff) {
    MultinomialLogisticLossBackwardGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(piece_size), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            piece_size, num_of_classes, prediction, labels, prediction_diff);
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(MultinomialLogisticLossKernelUtil);

}  // namespace oneflow
