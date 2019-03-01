#include "oneflow/core/kernel/sigmoid_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SigmoidKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  NewKernelUtil<device_type, T>::Sigmoid(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                         in_blob->dptr<T>(), BnInOp2Blob("out")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void SigmoidKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob("out");
  NewKernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), out_blob->dptr<T>(), out_blob->dptr<T>(),
      BnInOp2Blob("out_diff")->dptr<T>(), BnInOp2Blob("in_diff")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSigmoidConf, SigmoidKernel,
                           FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

}  // namespace oneflow
