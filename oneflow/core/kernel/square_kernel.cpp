#include "oneflow/core/kernel/square_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SquareKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
void SquareKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSquareConf, SquareKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
