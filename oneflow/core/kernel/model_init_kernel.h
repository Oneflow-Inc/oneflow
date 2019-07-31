#ifndef ONEFLOW_CORE_KERNEL_MODEL_INIT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_INIT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ModelInitKernel final : public KernelIf<device_type> {
 private:
  void VirtualKernelInit(const ParallelContext*) override;

  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

  int32_t part_id_ = -1;
  int32_t part_num_ = -1;
  std::unique_ptr<std::mt19937> random_seed_gen_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_INIT_KERNEL_H_
