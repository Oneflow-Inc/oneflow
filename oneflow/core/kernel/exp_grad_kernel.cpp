#include "oneflow/core/kernel/exp_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ExpGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* dy = BnInOp2Blob("dy");
  const Blob* y = BnInOp2Blob("y");
  Blob* dx = BnInOp2Blob("dx");
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, dy->shape().elem_cnt(), dy->dptr<T>(),
                                  y->dptr<T>(), dx->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ExpGradKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kExpGradConf, ExpGradKernel, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
