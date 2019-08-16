#ifndef ONEFLOW_CORE_KERNEL_SPLIT_LIKE_KERNEL_H
#define ONEFLOW_CORE_KERNEL_SPLIT_LIKE_KERNEL_H

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SplitLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SplitLikeKernel);
  SplitLikeKernel() = default;
  ~SplitLikeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SPLIT_LIKE_KERNEL_H
