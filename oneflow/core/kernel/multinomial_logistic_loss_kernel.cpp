#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void MultinomialLogisticLossKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* prediction = BnInOp2BlobPtr("prediction");
  const Blob* label = BnInOp2BlobPtr("label");
  Blob* loss = BnInOp2BlobPtr("loss");
  Blob* loss_buff = BnInOp2BlobPtr("loss_buffer");

  MultinomialLogisticLossKernelUtil<device_type, FloatingPointType>::Forward(
      ctx, prediction->shape().At(0), prediction->shape().At(1),
      prediction->dptr<FloatingPointType>(), label->dptr<FloatingPointType>(),
      loss->mut_dptr<FloatingPointType>(),
      loss_buff->mut_dptr<FloatingPointType>());

  Blob* prediction_diff = BnInOp2BlobPtr(GenDiffBn("prediction"));
  if (prediction_diff != nullptr) {
    MultinomialLogisticLossKernelUtil<device_type, FloatingPointType>::Backward(
        ctx, prediction->shape().At(0), prediction->shape().At(1),
        prediction->dptr<FloatingPointType>(), label->dptr<FloatingPointType>(),
        prediction_diff->mut_dptr<FloatingPointType>());
  }
}

template<typename FloatingPointType>
class MultinomialLogisticLossKernelUtil<DeviceType::kCPU, FloatingPointType>
    final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t instance_num,
                      const int64_t num_of_classes,
                      const FloatingPointType* prediction,
                      const FloatingPointType* labels, FloatingPointType* loss,
                      FloatingPointType* loss_buff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      loss[0] = 0;
      for (int64_t i = 0; i < instance_num; ++i) {
        int64_t label = labels[i];
        FloatingPointType prob = prediction[i * num_of_classes + label];
        loss[0] -= SAFE_LOG(prob);
      }
      loss[0] = loss[0] / instance_num;
    });
  }

  static void Backward(const KernelCtx& ctx, const int64_t instance_num,
                       const int64_t num_of_classes,
                       const FloatingPointType* prediction,
                       const FloatingPointType* labels,
                       FloatingPointType* prediction_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      const FloatingPointType scale = -1.0 / instance_num;
      for (int64_t i = 0; i < instance_num; i++) {
        int64_t label = labels[i];
        FloatingPointType prob =
            MAX_WITH_LOG_THRESHOLD(prediction[i * num_of_classes + label]);
        prediction_diff[i * num_of_classes + label] = scale / prob;
      }
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(MultinomialLogisticLossKernelUtil);
INSTANTIATE_KERNEL_CLASS(MultinomialLogisticLossKernel);
REGISTER_KERNEL(OperatorConf::kMultinomialLogisticLossConf,
                MultinomialLogisticLossKernel);

}  // namespace oneflow
