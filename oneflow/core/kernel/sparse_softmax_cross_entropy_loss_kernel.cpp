#include "oneflow/core/kernel/sparse_softmax_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SparseSoftmaxCrossEntropyLossKernel<device_type, PredType, LabelType>::
    VirtualLossForwardDataContent(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  Blob* prob_blob = BnInOp2Blob("prob");
  Blob* loss_blob = BnInOp2Blob("loss");
  Blob* buf_blob = BnInOp2Blob("fw_buf");
  const int64_t n = prediction_blob->shape().At(0);
  const int64_t w = prediction_blob->shape().Count(1);
  const PredType* pred = prediction_blob->dptr<PredType>();
  const LabelType* label = label_blob->dptr<LabelType>();
  PredType* prob = prob_blob->mut_dptr<PredType>();
  PredType* loss = loss_blob->mut_dptr<PredType>();
  // forward
  SoftmaxComputeProb<device_type, PredType>(ctx.device_ctx, n, w, pred, loss, prob,
                                            buf_blob->mut_dptr(),
                                            buf_blob->ByteSizeOfDataContentField());
  SparseCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Forward(ctx.device_ctx, n, w,
                                                                              prob, label, loss);
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf&
SparseSoftmaxCrossEntropyLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.sparse_softmax_cross_entropy_loss_conf().loss_conf();
}

template<typename PredType, typename LabelType>
struct SparseSoftmaxCrossEntropyLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w, const LabelType* label,
                          PredType* in_diff) {
    for (int64_t i = 0; i < n; ++i) { in_diff[i * w + static_cast<int64_t>(label[i])] -= 1; }
  }
};

namespace {

Kernel* CreateSparseSoftmaxCrossEntropyLossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_KERNEL_ENTRY(device_type, pred_type_pair,                \
                                                       label_type_pair)                            \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new SparseSoftmaxCrossEntropyLossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair), \
                                                    OF_PP_PAIR_FIRST(label_type_pair)>();          \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_KERNEL_ENTRY,
                                       DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                 kernel_conf.sparse_softmax_cross_entropy_loss_conf().loss_conf().prediction_type(),
                 kernel_conf.sparse_softmax_cross_entropy_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSparseSoftmaxCrossEntropyLossConf,
                        CreateSparseSoftmaxCrossEntropyLossKernel);

}  // namespace oneflow
