#include "oneflow/core/kernel/gather_ms0_kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& GatherMs0Kernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_ms0_conf();
}

template<DeviceType device_type, typename T>
void GatherMs0Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const int64_t offset = this->kernel_conf().gather_ms0_conf().offset();
  GatherKernelUtil<device_type, T>::Forward(ctx.device_ctx, indices, in, 0, out, offset);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherMs0Conf, GatherMs0Kernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
