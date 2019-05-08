#include "oneflow/core/kernel/identity_kernel.h"

namespace oneflow {

namespace {

void SafeCopyBlob(DeviceCtx* ctx, Blob* dst, const Blob* src) {
  CHECK_EQ(src->ByteSizeOfDataContentField(), dst->ByteSizeOfDataContentField());
  dst->CopyDataContentFrom(ctx, src);
}

}  // namespace

template<DeviceType device_type>
void IdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  SafeCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
}

template<DeviceType device_type>
void IdentityKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  SafeCopyBlob(ctx.device_ctx, BnInOp2Blob(GenDiffBn("in")), BnInOp2Blob(GenDiffBn("out")));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kIdentityConf, IdentityKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kParallelCastConf, IdentityKernel);

}  // namespace oneflow
