#ifndef ONEFLOW_CORE_KERNEL_LOCAL_GPU_PEER_PARTIAL_SUM_TO_SPLIT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOCAL_GPU_PEER_PARTIAL_SUM_TO_SPLIT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

template<typename T>
class LocalGpuPeerPartialSumToSplitKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGpuPeerPartialSumToSplitKernel);
  LocalGpuPeerPartialSumToSplitKernel() = default;
  ~LocalGpuPeerPartialSumToSplitKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOCAL_GPU_PEER_PARTIAL_SUM_TO_SPLIT_KERNEL_H_
