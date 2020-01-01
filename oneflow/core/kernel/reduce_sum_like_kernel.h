#ifndef ONEFLOW_CORE_KERNEL_REDUCE_SUM_LIKE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REDUCE_SUM_LIKE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ReduceSumLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSumLikeKernel);
  ReduceSumLikeKernel() = default;
  ~ReduceSumLikeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_SUM_LIKE_KERNEL_H_
