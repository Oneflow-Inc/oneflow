#ifndef ONEFLOW_CORE_KERNEL_MODEL_SAVE_V2_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_SAVE_V2_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class ModelSaveV2Kernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveV2Kernel);
  ModelSaveV2Kernel() = default;
  ~ModelSaveV2Kernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_SAVE_V2_KERNEL_H_
