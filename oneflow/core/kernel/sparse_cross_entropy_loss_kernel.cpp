#include "oneflow/core/kernel/sparse_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SparseCrossEntropyLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* loss = BnInOp2Blob("loss");

  SparseCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, prediction->shape().At(0), prediction->shape().At(1),
      prediction->dptr<PredType>(), label->dptr<LabelType>(), loss->mut_dptr<PredType>());
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf&
SparseCrossEntropyLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.sparse_cross_entropy_loss_conf().loss_conf();
}

template<typename PredType, typename LabelType>
struct SparseCrossEntropyLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t instance_num, const int64_t num_of_classes,
                      const PredType* prediction, const LabelType* labels, PredType* loss) {
    for (int64_t i = 0; i < instance_num; ++i) {
      int64_t label = static_cast<int64_t>(labels[i]);
      CHECK_GE(label, 0);
      CHECK_LT(label, num_of_classes);
      loss[i] = -SafeLog(prediction[i * num_of_classes + label]);
    }
  }

  static void Backward(DeviceCtx* ctx, const int64_t instance_num, const int64_t num_of_classes,
                       const PredType* prediction, const LabelType* labels,
                       PredType* prediction_diff) {
    for (int64_t i = 0; i < instance_num; ++i) {
      int64_t label = static_cast<int64_t>(labels[i]);
      PredType prob = MaxWithLogThreshold(prediction[i * num_of_classes + label]);
      prediction_diff[i * num_of_classes + label] = -1 / prob;
    }
  }
};

namespace {

Kernel* CreateSparseCrossEntropyLossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SPARSE_CROSS_ENTROPY_LOSS_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair)       \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new SparseCrossEntropyLossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),        \
                                             OF_PP_PAIR_FIRST(label_type_pair)>();                 \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SPARSE_CROSS_ENTROPY_LOSS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                 kernel_conf.sparse_cross_entropy_loss_conf().loss_conf().prediction_type(),
                 kernel_conf.sparse_cross_entropy_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSparseCrossEntropyLossConf,
                        CreateSparseCrossEntropyLossKernel);

#define MAKE_ENTRY(data_type_pair, label_type_pair) \
  template struct SparseCrossEntropyLossKernelUtil< \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
