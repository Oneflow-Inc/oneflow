#include "oneflow/core/kernel/broadcast_add_kernel.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BroadcastBinaryKernelUtil<device_type, T, BinaryFuncAdd>::Forward(kernel_ctx, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void BroadcastAddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastAddConf, BroadcastAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
