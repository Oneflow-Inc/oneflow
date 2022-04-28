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
#include <vector>
#include "oneflow/core/device/npu_event.h"

namespace oneflow {

#ifdef WITH_NPU

NpuEvent::NpuEvent(int device_id, unsigned int flags) : device_id_(device_id) {
  std::cout<<"NpuEvent::NpuEvent"<<std::endl;
  NpuCurrentDeviceGuard guard(device_id_);
  OF_NPU_CHECK(aclrtCreateEventWithFlag(&event_, flags));
}

NpuEvent::~NpuEvent() {
  std::cout<<"NpuEvent::~NpuEvent"<<std::endl;
  NpuCurrentDeviceGuard guard(device_id_);
  OF_NPU_CHECK(aclrtDestroyEvent(event_));
}

bool NpuEvent::Query() const { 
    aclrtEventStatus status;
    aclrtQueryEvent(event_,&status);
    return  status!= ACL_EVENT_STATUS_NOT_READY;
}//dck_caution_here

#endif

}  // namespace oneflow
