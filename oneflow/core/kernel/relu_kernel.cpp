#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* x_blob = BnInOp2Blob("x");
  KernelUtil<device_type, T>::Relu(ctx.device_ctx, x_blob->shape().elem_cnt(), x_blob->dptr<T>(),
                                   BnInOp2Blob("y")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* y_blob = BnInOp2Blob("y");
  KernelUtil<device_type, T>::ReluBackward(
      ctx.device_ctx, y_blob->shape().elem_cnt(), y_blob->dptr<T>(), y_blob->dptr<T>(),
      BnInOp2Blob("y_diff")->dptr<T>(), BnInOp2Blob("x_diff")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReluConf, ReluKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
