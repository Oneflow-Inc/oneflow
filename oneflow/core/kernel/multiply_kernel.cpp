#include "oneflow/core/kernel/multiply_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MultiplyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  // out = in .* weight
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                  weight_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void MultiplyKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  // weight_diff = out_diff * in
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                  out_diff_blob->dptr<T>(), weight_diff_blob->mut_dptr<T>());
  // in_diff = out_diff * weight
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, weight_blob->shape().elem_cnt(),
                                  weight_blob->dptr<T>(), out_diff_blob->dptr<T>(),
                                  in_diff_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
const PbMessage& MultiplyKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().multiply_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMultiplyConf, MultiplyKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
