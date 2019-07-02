#ifndef ONEFLOW_CORE_KERNEL_MULTI_RING_ALL_REDUCE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MULTI_RING_ALL_REDUCE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MultiRingAllReduceKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiRingAllReduceKernel);
  MultiRingAllReduceKernel() = default;
  ~MultiRingAllReduceKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  int64_t num_steps_ = -1;
  std::vector<std::vector<Range>> chunk_slices_;
};

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_KERNEL_MULTI_RING_ALL_REDUCE_KERNEL_H_
