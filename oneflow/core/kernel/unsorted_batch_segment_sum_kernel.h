#ifndef ONEFLOW_CORE_KERNEL_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class UnsortedBatchSegmentSumKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnsortedBatchSegmentSumKernel);
  UnsortedBatchSegmentSumKernel() = default;
  ~UnsortedBatchSegmentSumKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_H_
