#ifndef ONEFLOW_CORE_KERNEL_WAIT_AND_SEND_IDS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_WAIT_AND_SEND_IDS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/buffer_manager.h"

namespace oneflow {

struct WaitAndSendIdsStatus final {
  ChannelStatus channel_status_;
  int64_t in_id_;
  int64_t out_idx_;
  size_t out_num_;
};

template<typename T>
class WaitAndSendIdsKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WaitAndSendIdsKernel);
  WaitAndSendIdsKernel() = default;
  ~WaitAndSendIdsKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_WAIT_AND_SEND_IDS_KERNEL_H_
