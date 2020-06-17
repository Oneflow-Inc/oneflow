#ifndef ONEFLOW_CORE_DEVICE_COLLECTIVE_BOXING_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_COLLECTIVE_BOXING_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CollectiveBoxingDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingDeviceCtx);
  CollectiveBoxingDeviceCtx() = default;
  ~CollectiveBoxingDeviceCtx() override = default;

  void SetCheckPoint(std::shared_ptr<std::atomic<bool>> ready_flag);
  void AddCallBack(std::function<void()> callback) const override;

 private:
  std::shared_ptr<std::atomic<bool>> current_ready_flag_;

};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_COLLECTIVE_BOXING_DEVICE_CONTEXT_H_
