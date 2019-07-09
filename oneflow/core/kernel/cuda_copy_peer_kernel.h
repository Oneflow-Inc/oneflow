#ifndef ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class CudaCopyPeerKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaCopyPeerKernel);
  CudaCopyPeerKernel() = default;
  ~CudaCopyPeerKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_H_
