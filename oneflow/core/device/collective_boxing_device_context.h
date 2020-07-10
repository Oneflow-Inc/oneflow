#ifndef ONEFLOW_CORE_DEVICE_COLLECTIVE_BOXING_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_COLLECTIVE_BOXING_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/job/collective_boxing_device_ctx_poller.h"

namespace oneflow {

using namespace boxing::collective;

class CollectiveBoxingDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingDeviceCtx);
  CollectiveBoxingDeviceCtx() = default;
  ~CollectiveBoxingDeviceCtx() override = default;

  std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> AddCheckpoint();
  void AddCallBack(std::function<void()> callback) const override;

 private:
  std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> current_checkpoint_;

};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_COLLECTIVE_BOXING_DEVICE_CONTEXT_H_
