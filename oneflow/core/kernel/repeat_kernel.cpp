#include "oneflow/core/kernel/repeat_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RepeatKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

template<DeviceType device_type, typename T>
void RepeatKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  /*
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));

  const int64_t acc_count = *static_cast<int64_t*>(ctx.other);

  if (acc_count == 0) {
    in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  } else {
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, out_diff_blob->shape().elem_cnt(), 1.0,
                                     out_diff_blob->dptr<T>(), 1, in_diff_blob->mut_dptr<T>(), 1);
  }
  */
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRepeatConf, RepeatKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
