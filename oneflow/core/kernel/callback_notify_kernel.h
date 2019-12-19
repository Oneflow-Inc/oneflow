#ifndef ONEFLOW_CORE_KERNEL_CALLBACK_NOTIFY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CALLBACK_NOTIFY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/buffer_manager.h"

namespace oneflow {

template<typename T>
class CallbackNotifyKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CallbackNotifyKernel);
  CallbackNotifyKernel() = default;
  ~CallbackNotifyKernel() = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CALLBACK_NOTIFY_KERNEL_H_
