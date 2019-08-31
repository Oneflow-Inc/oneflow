#include "oneflow/core/kernel/identity_kernel.h"

namespace oneflow {

namespace {

void CheckSizeAndCopyBlob(DeviceCtx *ctx, Blob *dst, const Blob *src) {
  CHECK_EQ(src->ByteSizeOfDataContentField(), dst->ByteSizeOfDataContentField());
  dst->CopyDataContentFrom(ctx, src);
}

}  // namespace

template<DeviceType device_type>
void IdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kIdentityConf, DeviceType::kCPU,
                            IdentityKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kIdentityConf, DeviceType::kGPU,
                            IdentityKernel<DeviceType::kGPU>);

}  // namespace oneflow
