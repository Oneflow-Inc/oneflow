#include "oneflow/core/kernel/square_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SquareKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<device_type, T>::Square(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                     in_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSquareConf, SquareKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
