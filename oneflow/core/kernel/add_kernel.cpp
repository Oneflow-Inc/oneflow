#include "oneflow/core/kernel/add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* in_blob_0 = BnInOp2Blob(this->kernel_conf().input_bns(0));
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob_0);
  const int64_t elem_cnt = out_blob->shape().elem_cnt();
  FOR_RANGE(size_t, i, 1, this->kernel_conf().input_bns().size()) {
    const Blob* in_blob = BnInOp2Blob(this->kernel_conf().input_bns(i));
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt,
                                     static_cast<T>(1), in_blob->dptr<T>(), 1,
                                     out_blob->mut_dptr<T>(), 1);
  }
}

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  FOR_RANGE(size_t, i, 0, this->kernel_conf().input_diff_bns().size()) {
    Blob* in_diff_blob = BnInOp2Blob(this->kernel_conf().input_diff_bns(i));
    in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAddConf, AddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
