#include "oneflow/core/kernel/identity_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType>
void IdentityLossKernel<device_type, PredType>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf initializer;
  initializer.mutable_constant_conf()->set_value(1.0);
  KernelUtil<device_type, PredType>::InitializeWithConf(ctx, initializer, 0, BnInOp2Blob("ones"));
}

template<DeviceType device_type, typename PredType>
void IdentityLossKernel<device_type, PredType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* ones = BnInOp2Blob("ones");
  Blob* loss = BnInOp2Blob("loss");
  Blob* prediction_diff = BnInOp2Blob(GenDiffBn("prediction"));
  loss->CopyDataContentFrom(ctx.device_ctx, prediction);
  if (prediction_diff != nullptr) { prediction_diff->CopyDataContentFrom(ctx.device_ctx, ones); }
}

template<DeviceType device_type, typename PredType>
const LossKernelConf& IdentityLossKernel<device_type, PredType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.identity_loss_conf().loss_conf();
}

template<DeviceType device_type, typename PredType>
int64_t IdentityLossKernel<device_type, PredType>::CalcLossInstanceNum(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  return BnInOp2Blob("prediction")->shape().elem_cnt();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kIdentityLossConf, IdentityLossKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
