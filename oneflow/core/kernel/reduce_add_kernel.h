#ifndef ONEFLOW_CORE_KERNEL_REDUCE_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REDUCE_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ReduceAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceAddKernel);
  ReduceAddKernel() = default;
  ~ReduceAddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  bool HasEmptyShapeBlob(
      const KernelCtx& ctx, const PbRpf<std::string>& bns,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_ADD_KERNEL_H_
