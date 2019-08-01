#include "oneflow/core/kernel/variable_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void VariableKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ++(*tick_);
}

template<DeviceType device_type, typename T>
const PbMessage& VariableKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().variable_conf();
}

template<DeviceType device_type, typename T>
const std::string& VariableKernel<device_type, T>::ModelName() const {
  return this->op_conf().variable_conf().model_name();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kVariableConf, VariableKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
