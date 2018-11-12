#include "oneflow/core/kernel/const_range_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConstRangeKernel<device_type, T>::VirtualKernelInit(const ParallelContext*) {}

template<DeviceType device_type, typename T>
void ConstRangeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

template<DeviceType device_type, typename T>
const PbMessage& ConstRangeKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().const_range_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConstRangeConf, ConstRangeKernel, INT_DATA_TYPE_SEQ);

}  // namespace oneflow
