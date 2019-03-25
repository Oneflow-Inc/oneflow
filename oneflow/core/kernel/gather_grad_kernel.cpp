#include "oneflow/core/kernel/gather_grad_kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& GatherGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_grad_conf();
}

template<DeviceType device_type, typename T>
void GatherGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  const Blob* out_diff = BnInOp2Blob("out_diff");
  Blob* in_diff = BnInOp2Blob("in_diff");
  Memset<device_type>(ctx.device_ctx, in_diff->mut_dptr<T>(), 0,
                      in_diff->ByteSizeOfDataContentField());
  GatherKernelUtil<device_type, T>::Backward(ctx.device_ctx, indices, out_diff,
                                             this->kernel_conf().gather_conf().axis(), in_diff);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherGradConf, GatherGradKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
