#ifndef ONEFLOW_CORE_KERNEL_MODEL_LOAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_LOAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ModelLoadKernel final : public KernelIf<device_type> {
 private:
  void VirtualKernelInit(const ParallelContext*) override;

  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

  int32_t part_id_ = -1;
  int32_t part_num_ = -1;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_LOAD_KERNEL_H_
