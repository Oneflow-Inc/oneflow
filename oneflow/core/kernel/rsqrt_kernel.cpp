#include "oneflow/core/kernel/rsqrt_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RsqrtKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

template<DeviceType device_type, typename T>
void RsqrtKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRsqrtConf, RsqrtKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
