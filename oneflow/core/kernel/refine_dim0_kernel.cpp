#include "oneflow/core/kernel/refine_dim0_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void RefineDim0Kernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type>
void RefineDim0Kernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob(GenDiffBn("in"))->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(GenDiffBn("out")));
}

template<DeviceType device_type>
void RefineDim0Kernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

template<DeviceType device_type>
void RefineDim0Kernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

template<DeviceType device_type>
void RefineDim0Kernel<device_type>::BackwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kRefineDim0Conf, RefineDim0Kernel);

}  // namespace oneflow
