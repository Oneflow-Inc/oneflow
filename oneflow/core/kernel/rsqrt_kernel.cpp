#include "oneflow/core/kernel/rsqrt_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
void RsqrtKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  KernelUtil<device_type, T>::Rsqrt(ctx.device_ctx, out->shape().elem_cnt(), in->dptr<T>(),
                                    out->mut_dptr<T>(), 0);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRsqrtConf, RsqrtKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace

}  // namespace oneflow
