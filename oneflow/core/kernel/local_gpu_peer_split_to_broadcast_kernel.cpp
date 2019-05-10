#include "oneflow/core/kernel/local_gpu_peer_split_to_broadcast_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

template<typename T>
void LocalGpuPeerSplitToBroadcastKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalGpuPeerSplitToBroadcastConf,
                               LocalGpuPeerSplitToBroadcastKernel, ARITHMETIC_DATA_TYPE_SEQ)

#endif

}  // namespace oneflow
