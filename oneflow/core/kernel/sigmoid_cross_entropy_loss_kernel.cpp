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
  Blob* loss_buf = BnInOp2Blob("loss_buf");
  Blob* tmp_storage = BnInOp2Blob("sum_buf");
  const size_t tmp_storage_byte_size = static_cast<size_t>(tmp_storage->shape().At(0));
  Blob* count = BnInOp2Blob("count");
  Blob* label_num = BnInOp2Blob("label_num");
  Blob* loss = BnInOp2Blob("loss");
  Blob* prediction_diff = BnInOp2Blob(GenDiffBn("prediction"));

  SigmoidCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, conf, prediction->shape().elem_cnt(), prediction->dptr<PredType>(),
      label->dptr<LabelType>(), loss_buf->mut_dptr<PredType>(), tmp_storage->mut_dptr<PredType>(),
      tmp_storage_byte_size, count->mut_dptr<PredType>(), label_num->mut_dptr<PredType>(),
      loss->mut_dptr<PredType>());

  if (prediction_diff != nullptr) {
    Memset<device_type>(ctx.device_ctx, prediction_diff->mut_dptr<PredType>(), 0,
                        prediction_diff->ByteSizeOfDataContentField());
    SigmoidCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Backward(
        ctx.device_ctx, conf, prediction->shape().elem_cnt(), prediction->dptr<PredType>(),
        label->dptr<LabelType>(), label_num->dptr<PredType>(),
        prediction_diff->mut_dptr<PredType>());
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
                      PredType* tmp_storage, const size_t tmp_storage_byte_size, PredType* count,
                      PredType* label_num, PredType* loss) {
    UNIMPLEMENTED();
  }

  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                       const PredType* prediction, const LabelType* label,
                       const PredType* label_num, PredType* pred_diff) {
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
