#include "oneflow/core/kernel/gather_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& GatherKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_conf();
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx&, std::function<Blob*(const std::string&)>) const {}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx&, std::function<Blob*(const std::string&)>) const {}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
