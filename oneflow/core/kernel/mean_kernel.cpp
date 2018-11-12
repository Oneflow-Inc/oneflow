#include "oneflow/core/kernel/mean_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MeanKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(1, out_blob->shape().elem_cnt());
  KernelUtil<device_type, T>::Sum(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                  out_blob->mut_dptr<T>(), fw_tmp_blob->mut_dptr<T>(),
                                  fw_tmp_blob->ByteSizeOfDataContentField());

  size_t total_elem_num = in_blob->shape().elem_cnt();
  KernelUtil<device_type, T>::Div(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  out_blob->mut_dptr<T>(), static_cast<T>(total_elem_num));
}

template<DeviceType device_type, typename T>
void MeanKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));

  size_t total_elem_num = in_diff_blob->shape().elem_cnt();
  KernelUtil<device_type, T>::Div(ctx.device_ctx, in_diff_blob->shape().elem_cnt(),
                                  in_diff_blob->mut_dptr<T>(), static_cast<T>(total_elem_num));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMeanConf, MeanKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
