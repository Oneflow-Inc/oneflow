#include "oneflow/core/device/collective_boxing_device_context.h"

namespace oneflow {

std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> CollectiveBoxingDeviceCtx::AddCheckpoint() {
  std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> checkpoint =
      Global<boxing::collective::CollectiveBoxingDeviceCtxPoller>::Get()->CreateCheckpoint();
  current_checkpoint_ = checkpoint;
  return checkpoint;
}

void CollectiveBoxingDeviceCtx::AddCallBack(std::function<void()> callback) const {
  CHECK(current_checkpoint_);
  Global<boxing::collective::CollectiveBoxingDeviceCtxPoller>::Get()->Enqueue(current_checkpoint_,
                                                                              callback);
}

}  // namespace oneflow
