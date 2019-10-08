#include "oneflow/core/kernel/output_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void OutputKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type>
void OutputKernel<device_type>::ForwardLoD(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->tree_lod_mut_view().UpdateLoD(BnInOp2Blob("in")->tree_lod_view().lod_tree());
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kOutputConf, OutputKernel);

}  // namespace oneflow
