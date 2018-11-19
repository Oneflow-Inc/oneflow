#include "oneflow/core/kernel/rsqrt_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RsqrtKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  TODO();
}

template<DeviceType device_type, typename T>
void RsqrtKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  TODO();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRsqrtConf, RsqrtKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
