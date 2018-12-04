#include "oneflow/core/kernel/scalar_mul_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ScalarMulKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), in_blob->dptr<T>(),
                      out_blob->ByteSizeOfDataContentField());
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                   static_cast<T>(this->op_conf().scalar_mul_conf().scalar()),
                                   out_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void ScalarMulKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  Memcpy<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), out_diff_blob->dptr<T>(),
                      out_diff_blob->ByteSizeOfDataContentField());
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, in_diff_blob->shape().elem_cnt(),
                                   static_cast<T>(this->op_conf().scalar_mul_conf().scalar()),
                                   in_diff_blob->mut_dptr<T>(), 1);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kScalarMulConf, ScalarMulKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
