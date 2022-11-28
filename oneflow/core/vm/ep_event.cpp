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
#include "oneflow/core/vm/ep_event.h"

namespace oneflow {

EpEvent::EpEvent(ep::Device* device) : device_(device), event_(nullptr) {
  device_->SetAsActiveDevice();
  event_ = device_->CreateEvent();  // NOLINT
}

EpEvent::~EpEvent() {
  device_->SetAsActiveDevice();
  device_->DestroyEvent(event_);
}

bool EpEvent::Query() const {
  device_->SetAsActiveDevice();
  return CHECK_JUST(event_->QueryDone());
}

}  // namespace oneflow
