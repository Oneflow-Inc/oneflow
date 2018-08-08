#include "oneflow/core/kernel/sigmoid_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SigmoidCrossEntropyLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const SigmoidCrossEntropyLossOpConf& conf = this->op_conf().sigmoid_cross_entropy_loss_conf();
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  // average_loss is the final result
  Blob* loss_buf = BnInOp2Blob("loss_buf");
  Blob* loss = BnInOp2Blob("loss");
  Blob* count = BnInOp2Blob("count");
  Blob* normalize = BnInOp2Blob("normalize");
  Blob* prediction_diff = BnInOp2Blob(GenDiffBn("prediction"));

  SigmoidCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, conf, prediction->shape().At(0), prediction->dptr<PredType>(),
      label->dptr<LabelType>(), loss_buf->mut_dptr<PredType>(), count->mut_dptr<PredType>(),
      normalize->mut_dptr<PredType>(), loss->mut_dptr<PredType>());

  if (prediction_diff != nullptr) {
    Memset<device_type>(ctx.device_ctx, prediction_diff->mut_dptr<PredType>(), 0,
                        prediction_diff->ByteSizeOfDataContentField());
    SigmoidCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Backward(
        ctx.device_ctx, conf, prediction->shape().At(0), prediction->dptr<PredType>(),
        label->dptr<LabelType>(), prediction_diff->mut_dptr<PredType>(),
        normalize->mut_dptr<PredType>());
  }
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf&
SigmoidCrossEntropyLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.sigmoid_cross_entropy_loss_conf().loss_conf();
}

template<typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                      const PredType* prediction, const LabelType* label, PredType* loss_buf,
                      PredType* count, PredType* normalize, PredType* loss) {
    UNIMPLEMENTED();
  }

  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                       const PredType* prediction, const LabelType* label, PredType* pred_diff,
                       const PredType* normalize) {
    UNIMPLEMENTED();
  }
};

namespace {

Kernel* CreateSigmoidCrossEntropyLossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair)      \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new SigmoidCrossEntropyLossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),       \
                                              OF_PP_PAIR_FIRST(label_type_pair)>();                \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                 kernel_conf.sigmoid_cross_entropy_loss_conf().loss_conf().prediction_type(),
                 kernel_conf.sigmoid_cross_entropy_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSigmoidCrossEntropyLossConf,
                        CreateSigmoidCrossEntropyLossKernel);

}  // namespace oneflow
