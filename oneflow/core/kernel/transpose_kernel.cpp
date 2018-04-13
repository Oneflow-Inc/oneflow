#include "oneflow/core/kernel/transpose_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TransposeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Transpose<device_type, T>(ctx.device_ctx, BnInOp2Blob("in"),
                            BnInOp2Blob("out"),
                            this->kernel_conf().transpose_conf().perm());
}

template<DeviceType device_type, typename T>
void TransposeKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff = BnInOp2Blob("in_diff");
  if (in_diff) {
    Transpose<device_type, T>(
        ctx.device_ctx, BnInOp2Blob("out_diff"), in_diff,
        this->kernel_conf().transpose_conf().invert_perm());
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTransposeConf, TransposeKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
