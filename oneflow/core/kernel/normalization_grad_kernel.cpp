#include "oneflow/core/kernel/normalization_grad_kernel.h"
#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalizationGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NormalizationGradOpConf& conf = this->op_conf().normalization_grad_conf();
  NormalizationKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, BnInOp2Blob("x"), BnInOp2Blob("gamma"), BnInOp2Blob("mean"),
      BnInOp2Blob("inv_variance"), BnInOp2Blob("dy"), BnInOp2Blob("dx"), BnInOp2Blob("gamma_diff"),
      BnInOp2Blob("beta_diff"), BnInOp2Blob("buf"), conf.axis(), conf.epsilon());
}

template<DeviceType device_type, typename T>
const PbMessage& NormalizationGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().normalization_grad_conf();
}

template<DeviceType device_type, typename T>
void NormalizationGradKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& const_buf_bns = this->op_attribute().const_buf_bns();
  const auto ConstBnExists = [&](const std::string& bn) {
    return std::find(const_buf_bns.cbegin(), const_buf_bns.cend(), bn) != const_buf_bns.cend();
  };
  if (ConstBnExists("gamma")) {
    InitializerConf initializer;
    initializer.mutable_constant_conf()->set_value(1.0);
    KernelUtil<device_type, T>::InitializeWithConf(ctx, initializer, 0, BnInOp2Blob("gamma"));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationGradConf, NormalizationGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
