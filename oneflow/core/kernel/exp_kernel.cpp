#include "oneflow/core/kernel/exp_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ExpKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  CHECK_EQ(out->shape(), in->shape());
  KernelUtil<device_type, T>::Exp(ctx.device_ctx, in->shape().elem_cnt(), in->dptr<T>(),
                                  out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ExpKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kExpConf, ExpKernel, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
