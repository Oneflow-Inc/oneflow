#include "oneflow/core/kernel/add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* in_blob0 = BnInOp2Blob(this->kernel_conf().input_bns()[0]);
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob0);
  const int64_t count = out_blob->shape().elem_cnt();
  for (size_t i = 1; i < this->kernel_conf().input_bns().size(); ++i) {
    const Blob* in_blob = BnInOp2Blob(this->kernel_conf().input_bns()[i]);
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, count, static_cast<T>(1),
                                     in_blob->dptr<T>(), 1,
                                     out_blob->mut_dptr<T>(), 1);
  }
}

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  for (size_t i = 0; i < this->kernel_conf().input_diff_bns().size(); ++i) {
    Blob* in_diff_blob = BnInOp2Blob(this->kernel_conf().input_diff_bns()[i]);
    in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAddConf, AddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
