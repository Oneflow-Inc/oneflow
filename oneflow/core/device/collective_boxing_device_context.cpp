/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/device/collective_boxing_device_context.h"

namespace oneflow {

std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> CollectiveBoxingDeviceCtx::AddCheckpoint() {
  std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> checkpoint =
      Global<boxing::collective::CollectiveBoxingDeviceCtxPoller>::Get()->CreateCheckpoint();
  current_checkpoint_ = checkpoint;
  return checkpoint;
}

void CollectiveBoxingDeviceCtx::AddCallBack(std::function<void()> callback) const {
  Global<boxing::collective::CollectiveBoxingDeviceCtxPoller>::Get()->Enqueue(current_checkpoint_,
                                                                              callback);
}

}  // namespace oneflow
