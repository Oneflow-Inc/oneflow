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
#include "oneflow/core/lazy/stream_context/include/generic_stream_context.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/ep/include/active_device_guard.h"

namespace oneflow {

GenericStreamContext::GenericStreamContext(const StreamId& stream_id) : stream_(nullptr) {
  device_ =
      std::dynamic_pointer_cast<ep::Device>(Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(
          stream_id.device_type(), stream_id.device_index()));
  CHECK(device_);
  ep::ActiveDeviceGuard guard(device_.get());
  stream_ = dynamic_cast<ep::Stream*>(device_->CreateStream());
  CHECK(stream_ != nullptr);
  poller_thread_ = std::thread([this]() {
    CHECK_JUST(stream_->OnExecutionContextSetup());
    std::pair<ep::Event*, std::function<void()>> cb_event;
    while (cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      CHECK_JUST(cb_event.first->Sync());
      cb_event.second();
      device_->DestroyEvent(cb_event.first);
    }
    CHECK_JUST(stream_->OnExecutionContextTeardown());
  });
}

GenericStreamContext::~GenericStreamContext() {
  ep::ActiveDeviceGuard guard(device_.get());
  cb_event_chan_.Close();
  poller_thread_.join();
  device_->DestroyStream(stream_);
}

Maybe<void> GenericStreamContext::AddCallback(std::function<void()> callback) {
  ep::Event* event = device_->CreateEvent();
  stream_->RecordEvent(event);
  cb_event_chan_.Send(std::make_pair(event, std::move(callback)));
  return Maybe<void>::Ok();
}

ep::Stream* GenericStreamContext::stream() { return stream_; }

}  // namespace oneflow
