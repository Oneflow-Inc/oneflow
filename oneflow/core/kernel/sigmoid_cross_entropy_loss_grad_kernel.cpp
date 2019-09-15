#include "oneflow/core/kernel/sigmoid_cross_entropy_loss_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SigmoidCrossEntropyLossGradKernel<device_type, PredType, LabelType>::
    VirtualLossForwardDataContent(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const SigmoidCrossEntropyLossGradOpConf& conf =
      this->op_conf().sigmoid_cross_entropy_loss_grad_conf();

  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* pred_diff = BnInOp2Blob("prediction_diff");

  SigmoidCrossEntropyLossGradKernelUtil<device_type, PredType, LabelType>::Backward(
      ctx.device_ctx, conf, prediction->shape().elem_cnt(), prediction->dptr<PredType>(),
      label->dptr<LabelType>(), pred_diff->mut_dptr<PredType>());
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf&
SigmoidCrossEntropyLossGradKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.sigmoid_cross_entropy_loss_grad_conf().loss_conf();
}

template<typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossGradKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossGradOpConf& conf,
                       const int64_t n, const PredType* prediction, const LabelType* label,
                       PredType* pred_diff) {
    FOR_RANGE(int64_t, index, 0, n) {
      if (label[index] != -1) {
        pred_diff[index] = 1.f / (1.f + expf(-prediction[index])) - label[index];
      }
    }
  }
};

namespace {

Kernel* CreateSigmoidCrossEntropyLossGradKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SIGMOID_CROSS_ENTROPY_LOSS_GRAD_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair) \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new SigmoidCrossEntropyLossGradKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),   \
                                                  OF_PP_PAIR_FIRST(label_type_pair)>();            \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SIGMOID_CROSS_ENTROPY_LOSS_GRAD_KERNEL_ENTRY,
                                       DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                 kernel_conf.sigmoid_cross_entropy_loss_grad_conf().loss_conf().prediction_type(),
                 kernel_conf.sigmoid_cross_entropy_loss_grad_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSigmoidCrossEntropyLossGradConf,
                        CreateSigmoidCrossEntropyLossGradKernel);

}  // namespace oneflow
