#include "oneflow/core/kernel/identity_kernel.h"

namespace oneflow {

namespace {

void CheckSizeAndCopyBlob(DeviceCtx *ctx, Blob *dst, const Blob *src) {
  CHECK_EQ(src->ByteSizeOfBlobBody(), dst->ByteSizeOfBlobBody());
  dst->CopyDataContentFrom(ctx, src);
}

}  // namespace

template<DeviceType device_type>
void IdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
}

template<DeviceType device_type>
void IdentityKernel<device_type>::ForwardLoD(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  const Blob *in_blob = BnInOp2Blob("in");
  Blob *out_blob = BnInOp2Blob("out");
  out_blob->tree_lod_mut_view().UpdateLoD(in_blob->tree_lod_view().lod_tree());
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kIdentityConf, IdentityKernel);

}  // namespace oneflow
