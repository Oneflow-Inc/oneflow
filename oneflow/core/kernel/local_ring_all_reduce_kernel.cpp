#include "oneflow/core/kernel/local_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void LocalRingAllReduceKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalRingAllReduceConf, LocalRingAllReduceKernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
