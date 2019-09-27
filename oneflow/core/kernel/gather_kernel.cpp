#include "oneflow/core/kernel/gather_kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& GatherKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_conf();
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  GatherKernelUtil<device_type, T>::Forward(ctx.device_ctx, indices, in,
                                            this->kernel_conf().gather_conf().axis(), out);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
