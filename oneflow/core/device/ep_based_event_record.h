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
#ifndef ONEFLOW_CORE_DEVICE_EP_BASED_EVENT_RECORD_H_
#define ONEFLOW_CORE_DEVICE_EP_BASED_EVENT_RECORD_H_

#include "oneflow/core/device/event_record.h"
#include "oneflow/core/ep/include/active_device_guard.h"

namespace oneflow {

class EpBasedEventRecord : public EventRecord {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EpBasedEventRecord);
  EpBasedEventRecord(ep::Event* event, ep::Device* device) : event_(event), device_(device) {}
  ~EpBasedEventRecord() {
    ep::ActiveDeviceGuard guard(device_);
    device_->DestroyEvent(event_);
  };

  static std::shared_ptr<EventRecord> MakeEventRecord(ep::Stream* stream) {
    ep::Device* device = stream->device();
    ep::ActiveDeviceGuard guard(device);
    ep::Event* event = device->CreateEvent();
    stream->RecordEvent(event);
    return std::make_shared<EpBasedEventRecord>(event, device);
  }

  bool QueryDone() const override {
    ep::ActiveDeviceGuard guard(device_);
    bool done = CHECK_JUST(event_->QueryDone());
    return done;
  }

 private:
  ep::Event* event_;
  ep::Device* device_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_EP_BASED_EVENT_RECORD_H_
