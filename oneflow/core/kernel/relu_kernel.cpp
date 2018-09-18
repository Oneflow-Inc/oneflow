#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  KernelUtil<device_type, T>::Relu(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                   BnInOp2Blob("out")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<device_type, T>::ReluBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), out_blob->dptr<T>(), out_blob->dptr<T>(),
      BnInOp2Blob("out_diff")->dptr<T>(), BnInOp2Blob("in_diff")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReluConf, ReluKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
