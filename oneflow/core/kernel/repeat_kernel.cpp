#include "oneflow/core/kernel/repeat_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RepeatKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRepeatConf, RepeatKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
