#ifndef ONEFLOW_CORE_KERNEL_NCCL_ALL_GATHER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NCCL_ALL_GATHER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class NcclAllGatherKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclAllGatherKernel);
  NcclAllGatherKernel() = default;
  ~NcclAllGatherKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NCCL_ALL_GATHER_KERNEL_H_
