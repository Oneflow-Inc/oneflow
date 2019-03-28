#include "oneflow/core/kernel/normalization_grad_kernel.h"
#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalizationGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NormalizationGradOpConf& conf = this->op_conf().normalization_grad_conf();
  NormalizationKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("gamma"), BnInOp2Blob("mean"),
      BnInOp2Blob("inv_variance"), BnInOp2Blob("dy"), BnInOp2Blob("dx"), BnInOp2Blob("gamma_diff"),
      BnInOp2Blob("beta_diff"), BnInOp2Blob("buf"), conf.axis(), conf.epsilon());
}

template<DeviceType device_type, typename T>
void NormalizationGradKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type, typename T>
const PbMessage& NormalizationGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().normalization_grad_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationGradConf, NormalizationGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
