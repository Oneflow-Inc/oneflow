#include "oneflow/core/kernel/local_gpu_peer_partial_sum_to_broadcast_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

template<typename T>
void LocalGpuPeerPartialSumToBroadcastKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalGpuPeerPartialSumToBroadcastConf,
                               LocalGpuPeerPartialSumToBroadcastKernel, ARITHMETIC_DATA_TYPE_SEQ)

#endif

}  // namespace oneflow
