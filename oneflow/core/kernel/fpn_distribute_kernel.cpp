#include "oneflow/core/kernel/fpn_distribute_kernel.h"

namespace oneflow {

template<typename T>
void FpnDistributeKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnDistributeConf, FpnDistributeKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
