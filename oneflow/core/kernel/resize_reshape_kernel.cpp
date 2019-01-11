#include "oneflow/core/kernel/resize_reshape_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type>
void CopyWithTruncateOrPad(DeviceCtx* ctx, Blob* dst, const Blob* src) {
  const int64_t dst_size = dst->ByteSizeOfDataContentField();
  const int64_t src_size = src->ByteSizeOfDataContentField();
  Memcpy<device_type>(ctx, dst->mut_dptr(), src->dptr(), std::min(src_size, dst_size));
  if (dst_size > src_size) {
    Memset<device_type>(ctx, dst->mut_dptr<char>() + src_size, 0,
                        static_cast<size_t>(dst_size - src_size));
  }
}

}  // namespace

template<DeviceType device_type>
void ResizeReshapeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyWithTruncateOrPad<device_type>(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
}

template<DeviceType device_type>
void ResizeReshapeKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyWithTruncateOrPad<device_type>(ctx.device_ctx, BnInOp2Blob(GenDiffBn("in")),
                                     BnInOp2Blob(GenDiffBn("out")));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kResizeReshapeConf, ResizeReshapeKernel);

}  // namespace oneflow
