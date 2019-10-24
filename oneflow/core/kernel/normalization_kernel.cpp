#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& const_buf_bns = this->op_attribute().const_buf_bns();
  const auto ConstBnExists = [&](const std::string& bn) {
    return std::find(const_buf_bns.cbegin(), const_buf_bns.cend(), bn) != const_buf_bns.cend();
  };
  if (ConstBnExists("beta")) {
    InitializerConf initializer;
    initializer.mutable_constant_conf()->set_value(0);
    KernelUtil<device_type, T>::InitializeWithConf(ctx, initializer, 0, BnInOp2Blob("beta"));
  }
  if (ConstBnExists("gamma")) {
    InitializerConf initializer;
    initializer.mutable_constant_conf()->set_value(1.0);
    KernelUtil<device_type, T>::InitializeWithConf(ctx, initializer, 0, BnInOp2Blob("gamma"));
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NormalizationOpConf& conf = this->op_conf().normalization_conf();
  if (this->op_conf().trainable() && conf.is_training()) {
    NormalizationKernelUtil<device_type, T>::ForwardTraining(
        ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("gamma"), BnInOp2Blob("beta"),
        BnInOp2Blob("out"), BnInOp2Blob("moving_mean"), BnInOp2Blob("moving_variance"),
        BnInOp2Blob("mean"), BnInOp2Blob("inv_variance"), BnInOp2Blob("buf"), conf.axis(),
        conf.epsilon(), conf.momentum());
  } else {
    NormalizationKernelUtil<device_type, T>::ForwardInference(
        ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("gamma"), BnInOp2Blob("beta"),
        BnInOp2Blob("moving_mean"), BnInOp2Blob("moving_variance"), BnInOp2Blob("out"),
        BnInOp2Blob("buf"), conf.axis(), conf.epsilon());
  }
}

template<DeviceType device_type, typename T>
const PbMessage& NormalizationKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().normalization_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf, NormalizationKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
