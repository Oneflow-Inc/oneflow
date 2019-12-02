#include "oneflow/core/kernel/constant_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConstantKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (is_init_) { return; }
  CHECK(this->kernel_conf().has_constant_conf());
  const ConstantKernelConf& conf = this->kernel_conf().constant_conf();
  KernelUtil<device_type, T>::InitializeWithConf(ctx.device_ctx, conf.initializer(),
                                                 conf.random_seed(), BnInOp2Blob("out"));
  is_init_ = true;
}

template<DeviceType device_type, typename T>
const PbMessage& ConstantKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().constant_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConstantConf, ConstantKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
