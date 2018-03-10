#include "oneflow/core/kernel/maximum_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void MaximumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* in_blob0 = BnInOp2Blob(this->kernel_conf().input_bns()[0]);
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob0);
  Blob* mask_blob = BnInOp2Blob("mask");
  Memset<device_type>(ctx.device_ctx, mask_blob->mut_dptr(), 0,
                      mask_blob->ByteSizeOfDataContentField());
  const int count = out_blob->shape().elem_cnt();
  for (size_t i = 1; i < this->kernel_conf().input_bns().size(); ++i) {
    std::string& ibn = this->kernel_conf().input_bns()[i];
    const Blob* in_blob = BnInOp2Blob(ibn);
    KernelUtil<device_type, T>::ElementwiseMaxWithMask(
        ctx.device_ctx, count, out_blob->mut_dptr<T>(), in_blob->dptr<T>(), i,
        mask_blob->mut_dptr<int>());
  }
}

template<DeviceType device_type, typename T>
void MaximumKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* mask_blob = BnInOp2Blob("mask");
  Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  const int count = mask_blob->shape().elem_cnt();
  for (size_t i = 0; i < this->kernel_conf().input_diff_bns().size(); ++i) {
    const std::string& idbn = this->kernel_conf().input_diff_bns()[i];
    Blob* in_diff_blob = BnInOp2Blob(idbn);
    Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                        in_diff_blob->ByteSizeOfDataContentField());
    KernelUtil<device_type, T>::ElementwiseSetWithMask(
        ctx.device_ctx, count, in_diff_blob->mut_dptr<T>(),
        out_diff_blob->dptr<T>(), i, mask_blob->dptr<int>());
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaximumConf, MaximumKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
