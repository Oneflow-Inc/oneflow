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
#include "oneflow/core/thread/fake_device_thread.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/graph/id_serialization.h"

namespace oneflow {

FakeDeviceThread::FakeDeviceThread(int64_t thrd_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, thrd_id]() {
    OF_PROFILER_NAME_THIS_HOST_THREAD("Fake Device Actor : (" + std::to_string(thrd_id) + ")");
    ThreadCtx ctx;
#ifdef WITH_CUDA
    ctx.cb_event_chan = nullptr;
#endif  // WITH_CUDA
    PollMsgChannel(ctx);
  });
}

REGISTER_DEVICE_THREAD_CREATOR_WITH_STREAM_ID(DeviceType::kFAKEDEVICE,
                                              ([](const StreamId& stream_id) -> Thread* {
                                                return new FakeDeviceThread(
                                                    SerializeStreamIdToInt64(stream_id));
                                              }));

}  // namespace oneflow
