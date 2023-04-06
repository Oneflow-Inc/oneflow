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
#include "oneflow/core/ep/include/device.h"

namespace oneflow {

namespace ep {

Event* Device::CreateEvent() {
  Event* event = nullptr;
  this->CreateEvents(&event, 1);
  return event;
}

void Device::DestroyEvent(Event* event) { this->DestroyEvents(&event, 1); }

bool Device::IsStreamOrderedMemoryAllocationSupported() const { return false; }

}  // namespace ep

}  // namespace oneflow
