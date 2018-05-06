#ifndef ONEFLOW_CORE_KERNEL_REDUCE_GATHER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REDUCE_GATHER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class ReduceGatherKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGatherKernel);
  ReduceGatherKernel() = default;
  ~ReduceGatherKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  int32_t parallel_id_;
};

template<DeviceType device_type>
struct ReduceGatherKernelUtil {
  static void DoMemcpy(DeviceCtx* ctx, char* dst, const char* src, size_t sz,
                       bool is_same_parallel_id);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_GATHER_KERNEL_H_
