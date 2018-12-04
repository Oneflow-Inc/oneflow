#include "oneflow/core/kernel/rsqrt_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RsqrtKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<device_type, T>::Rsqrt(
      ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
      this->kernel_conf().op_attribute().op_conf().rsqrt_conf().epsilon(), out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void RsqrtKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  KernelUtil<device_type, T>::Powx(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                   out_blob->dptr<T>(), 3, in_diff_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                   static_cast<T>(-0.5), in_diff_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  out_diff_blob->dptr<T>(), in_diff_blob->dptr<T>(),
                                  in_diff_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRsqrtConf, RsqrtKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
