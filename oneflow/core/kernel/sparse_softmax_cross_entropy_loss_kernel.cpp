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

// SSCE: SparseSoftmaxCrossEntropy
#define REGISTER_SSCE_LOSS_KERNEL(pred_type, label_type)                                         \
  REGISTER_KERNEL_WITH_PRED_AND_LABEL(                                                           \
      OperatorConf::kSparseSoftmaxCrossEntropyLossConf, DeviceType::kGPU, pred_type, label_type, \
      SparseSoftmaxCrossEntropyLossKernel<DeviceType::kGPU, pred_type, label_type>)

REGISTER_SSCE_LOSS_KERNEL(float, int32_t);
REGISTER_SSCE_LOSS_KERNEL(float, int64_t);
REGISTER_SSCE_LOSS_KERNEL(double, int32_t);
REGISTER_SSCE_LOSS_KERNEL(double, int64_t);

}  // namespace oneflow
