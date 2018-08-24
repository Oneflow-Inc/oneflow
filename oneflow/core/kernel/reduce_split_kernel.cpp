#include "oneflow/core/kernel/reduce_split_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReduceSplitKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReduceSplitConf, ReduceSplitKernel);

}  // namespace oneflow
