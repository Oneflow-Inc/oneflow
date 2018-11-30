#include "oneflow/core/kernel/sparse_cross_entropy_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/sparse_cross_entropy_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SparseCrossEntropyKernel<device_type, PredType, LabelType>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* out_blob = BnInOp2Blob("out");

  SparseCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, prediction->shape().At(0), prediction->shape().At(1),
      prediction->dptr<PredType>(), label->dptr<LabelType>(), out_blob->mut_dptr<PredType>());
}

template<DeviceType device_type, typename PredType, typename LabelType>
void SparseCrossEntropyKernel<device_type, PredType, LabelType>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* prediction_diff = BnInOp2Blob(GenDiffBn("prediction"));
  Memset<device_type>(ctx.device_ctx, prediction_diff->mut_dptr<PredType>(), 0,
                      prediction_diff->ByteSizeOfDataContentField());
  SparseCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Backward(
      ctx.device_ctx, prediction->shape().At(0), prediction->shape().At(1),
      prediction->dptr<PredType>(), label->dptr<LabelType>(),
      prediction_diff->mut_dptr<PredType>());
}


}  // namespace oneflow
