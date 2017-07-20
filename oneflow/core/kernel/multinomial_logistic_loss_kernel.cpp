#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void MultinomialLogisticLossKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* prediction = BnInOp2BlobPtr("prediction");
  const Blob* label = BnInOp2BlobPtr("label");
  Blob* loss = BnInOp2BlobPtr("loss");

  MultinomialLogisticLossKernelUtil<device_type, FloatingPointType>::Forward(
      ctx,
      prediction->shape().At(0),  // piece size
      prediction->shape().At(1),  // number of classes
      prediction->dptr<FloatingPointType>(),
      label->dptr<FloatingPointType>(),
      loss->mut_dptr<FloatingPointType>());

  Blob* prediction_diff = BnInOp2BlobPtr("prediction_diff");
  if (prediction_diff != nullptr) { Backward(ctx, BnInOp2BlobPtr); }
}

template<DeviceType device_type, typename FloatingPointType>
void MultinomialLogisticLossKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* prediction = BnInOp2BlobPtr("prediction");
  const Blob* label = BnInOp2BlobPtr("label");
  Blob* prediction_diff = BnInOp2BlobPtr("prediction_diff");
  const Blob* loss_diff = BnInOp2BlobPtr("loss_diff");

  MultinomialLogisticLossKernelUtil<device_type, FloatingPointType>::Backward(
      ctx,
      prediction->shape().At(0),  // piece size
      prediction->shape().At(1),  // number of classes
      prediction->dptr<FloatingPointType>(),
      label->dptr<FloatingPointType>(),
      prediction_diff->mut_dptr<FloatingPointType>(),
      loss_diff->dptr<FloatingPointType>());
}

template<typename FloatingPointType>
class MultinomialLogisticLossKernelUtil<DeviceType::kCPU, FloatingPointType>
    final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t piece_size,
                      const int64_t num_of_classes,
                      const FloatingPointType* prediction,
                      const FloatingPointType* labels,
                      FloatingPointType* loss) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      loss[0] = 0;
      for (int64_t i = 0; i < piece_size; ++i) {
        int64_t label = labels[i];
        FloatingPointType prob =
            std::max(prediction[i * num_of_classes + label],
                     FloatingPointType(kLOG_THRESHOLD));
        loss[0] -= log(prob);
      }
      loss[0] = loss[0] / piece_size;
    });
  }

  static void Backward(const KernelCtx& ctx, const int64_t piece_size,
                       const int64_t num_of_classes,
                       const FloatingPointType* prediction,
                       const FloatingPointType* labels,
                       FloatingPointType* prediction_diff,
                       const FloatingPointType* loss_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      const FloatingPointType scale = -1.0 / piece_size;
      for (int64_t i = 0; i < piece_size; i++) {
        int64_t label = labels[i];
        FloatingPointType prob =
            std::max(prediction[i * num_of_classes + label],
                     FloatingPointType(kLOG_THRESHOLD));
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
