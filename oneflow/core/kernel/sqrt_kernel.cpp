#include "oneflow/core/kernel/sqrt_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SqrtKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<device_type, T>::Sqrt(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                   in_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSqrtConf, SqrtKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
