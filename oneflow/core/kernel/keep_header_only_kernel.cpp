#include "oneflow/core/kernel/keep_header_only_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void KeepHeaderOnlyKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type>
void KeepHeaderOnlyKernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type>
void KeepHeaderOnlyKernel<device_type>::ForwardDim2ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type>
void KeepHeaderOnlyKernel<device_type>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kKeepHeaderOnlyConf, KeepHeaderOnlyKernel);

}  // namespace oneflow
