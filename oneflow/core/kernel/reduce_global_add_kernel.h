#ifndef ONEFLOW_CORE_KERNEL_REDUCE_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REDUCE_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ReduceGlobalAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGlobalAddKernel);
  ReduceGlobalAddKernel() = default;
  ~ReduceGlobalAddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void VirtualKernelInit(const ParallelContext* parallel_ctx, DeviceCtx* device_ctx) override;

  int64_t parallel_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_ADD_KERNEL_H_
