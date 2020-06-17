#include "oneflow/core/device/collective_boxing_device_context.h"
#include "oneflow/core/job/collective_boxing_device_ctx_poller.h"

namespace oneflow {

void CollectiveBoxingDeviceCtx::SetCheckPoint(std::shared_ptr<std::atomic<bool>> ready_flag) {
  CHECK(ready_flag);
  current_ready_flag_ = std::move(ready_flag);
}

void CollectiveBoxingDeviceCtx::AddCallBack(std::function<void()> callback) const {
  CHECK(current_ready_flag_);
  Global<boxing::collective::CollectiveBoxingDeviceCtxPoller>::Get()->Enqueue(current_ready_flag_,
                                                                              callback);
}

}  // namespace oneflow
