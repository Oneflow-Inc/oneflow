#include "oneflow/core/kernel/tanh_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TanHKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  KernelUtil<device_type, T>::TanH(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                   BnInOp2Blob("out")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTanhConf, TanHKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
