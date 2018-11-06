#ifndef ONEFLOW_CORE_KERNEL_GROUP_BY_RECORD_ID_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_GROUP_BY_RECORD_ID_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class GroupByRecordIdKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GroupByRecordIdKernel);
  GroupByRecordIdKernel() = default;
  ~GroupByRecordIdKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim2ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_GROUP_BY_RECORD_ID_KERNEL_H_
