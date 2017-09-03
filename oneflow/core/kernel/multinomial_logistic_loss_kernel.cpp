#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void MultinomialLogisticLossKernel<device_type, PredType, LabelType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* prediction = BnInOp2BlobPtr("prediction");
  const Blob* label = BnInOp2BlobPtr("label");
  Blob* loss = BnInOp2BlobPtr("loss");
  Blob* loss_buff = BnInOp2BlobPtr("loss_buffer");

  MultinomialLogisticLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, prediction->shape().At(0), prediction->shape().At(1),
      prediction->dptr<PredType>(), label->dptr<LabelType>(),
      loss->mut_dptr<PredType>(), loss_buff->mut_dptr<PredType>());

  Blob* prediction_diff = BnInOp2BlobPtr(GenDiffBn("prediction"));
  if (prediction_diff != nullptr) {
    Memset<device_type>(ctx.device_ctx, prediction_diff->mut_dptr<PredType>(),
                        0, prediction_diff->TotalByteSize());
    MultinomialLogisticLossKernelUtil<device_type, PredType, LabelType>::
        Backward(ctx.device_ctx, prediction->shape().At(0),
                 prediction->shape().At(1), prediction->dptr<PredType>(),
                 label->dptr<LabelType>(),
                 prediction_diff->mut_dptr<PredType>());
  }
}

template<typename PredType, typename LabelType>
class MultinomialLogisticLossKernelUtil<DeviceType::kCPU, PredType, LabelType>
    final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const int64_t instance_num,
                      const int64_t num_of_classes, const PredType* prediction,
                      const LabelType* labels, PredType* loss,
                      PredType* loss_buff) {
    ctx->cpu_stream()->SendWork([=]() {
      loss[0] = 0;
      for (int64_t i = 0; i < instance_num; ++i) {
        PredType prob =
            prediction[i * num_of_classes + static_cast<int64_t>(labels[i])];
        loss[0] -= SAFE_LOG(prob);
      }
    });
  }

  static void Backward(DeviceCtx* ctx, const int64_t instance_num,
                       const int64_t num_of_classes, const PredType* prediction,
                       const LabelType* labels, PredType* prediction_diff) {
    ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < instance_num; ++i) {
        int64_t label = static_cast<int64_t>(labels[i]);
        PredType prob =
            MAX_WITH_LOG_THRESHOLD(prediction[i * num_of_classes + label]);
        prediction_diff[i * num_of_classes + label] = -1 / prob;
      }
    });
  }
};  // namespace oneflow

namespace {

template<DeviceType device_type>
Kernel* CreateMultinomialLogisticLossKernel(const OperatorConf& op_conf) {
  static const HashMap<std::string, std::function<Kernel*()>>
      data_type2creator = {
#define MULTI_LOG_LOSS_KERNEL_ENTRY(data_type_pair, label_type_pair) \
  {GetHashKey(OF_PP_PAIR_SECOND(data_type_pair),                     \
              OF_PP_PAIR_SECOND(label_type_pair)),                   \
   []() {                                                            \
     return new MultinomialLogisticLossKernel<                       \
         device_type, OF_PP_PAIR_FIRST(data_type_pair),              \
         OF_PP_PAIR_FIRST(label_type_pair)>;                         \
   }},
          OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MULTI_LOG_LOSS_KERNEL_ENTRY,
                                           FLOATING_DATA_TYPE_SEQ,
                                           INT_DATA_TYPE_SEQ)};
  return data_type2creator.at(GetHashKey(
      op_conf.multinomial_logistic_loss_conf().prediction().data_type(),
      op_conf.multinomial_logistic_loss_conf().label().data_type()))();
}

}  // namespace

REGISTER_TEMPLATE_KERNEL_CREATOR(OperatorConf::kMultinomialLogisticLossConf,
                                 CreateMultinomialLogisticLossKernel);

}  // namespace oneflow
