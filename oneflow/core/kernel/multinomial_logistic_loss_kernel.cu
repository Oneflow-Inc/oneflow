#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void MultinomialLogisticLossForwardGpu(
    const int64_t instance_num, const int64_t num_of_classes,
    const FloatingPointType* prediction, const FloatingPointType* labels,
    FloatingPointType* loss_buff) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    int64_t label = labels[i];
    FloatingPointType prob = prediction[i * num_of_classes + label];
    loss_buff[i] = -SAFE_LOG(prob) / instance_num;
  }
}

template<typename FloatingPointType>
__global__ void MultinomialLogisticLossBackwardGpu(
    const int64_t instance_num, const int64_t num_of_classes,
    const FloatingPointType* prediction, const FloatingPointType* labels,
    FloatingPointType* prediction_diff) {
  const FloatingPointType scale = -1.0 / instance_num;
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    int64_t label = labels[i];
    FloatingPointType prob = prediction[i * num_of_classes + label];
    prediction_diff[i * num_of_classes + label] =
        scale / MAX_WITH_LOG_THRESHOLD(prob);
  }
}

}  // namespace

template<typename FloatingPointType>
class MultinomialLogisticLossKernelUtil<DeviceType::kGPU, FloatingPointType>
    final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t instance_num,
                      const int64_t num_of_classes,
                      const FloatingPointType* prediction,
                      const FloatingPointType* labels, FloatingPointType* loss,
                      FloatingPointType* loss_buff) {
    MultinomialLogisticLossForwardGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(instance_num, num_of_classes,
                                            prediction, labels, loss_buff);
    KernelUtil<DeviceType::kGPU, FloatingPointType>::Sum(
        ctx, instance_num, loss_buff, loss, loss_buff,
        sizeof(FloatingPointType) * instance_num);
  }

  static void Backward(const KernelCtx& ctx, const int64_t instance_num,
                       const int64_t num_of_classes,
                       const FloatingPointType* prediction,
                       const FloatingPointType* labels,
                       FloatingPointType* prediction_diff) {
    MultinomialLogisticLossBackwardGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            instance_num, num_of_classes, prediction, labels, prediction_diff);
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(MultinomialLogisticLossKernelUtil);

}  // namespace oneflow
