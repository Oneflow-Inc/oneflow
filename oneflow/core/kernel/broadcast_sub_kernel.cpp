#include "oneflow/core/kernel/broadcast_sub_kernel.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastSubKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BroadcastBinaryKernelUtil<device_type, T, BinaryFuncSub>::Forward(kernel_ctx, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void BroadcastSubKernel<device_type, T>::BackwardDataContent(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastSubConf, BroadcastSubKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
