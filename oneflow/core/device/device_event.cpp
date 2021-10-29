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
#include "oneflow/core/device/device_event.h"

namespace oneflow {

#ifdef WITH_CUDA

DeviceEvent::DeviceEvent(int device_id, unsigned int flags) : device_id_(device_id) {
  CudaCurrentDeviceGuard guard(device_id_);
  OF_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

DeviceEvent::~DeviceEvent() {
  CudaCurrentDeviceGuard guard(device_id_);
  cudaEventDestroy(event_);
}

bool DeviceEvent::Query() const { return cudaEventQuery(event_) != cudaErrorNotReady; }

template<int device_id>
std::shared_ptr<DeviceEvent> GetReusedOrNewDeviceEvent(unsigned int flags) {
  using pool_type = obj_pool::ObjectPool<DeviceEvent, device_id + 1024>;
  DeviceEvent* ptr = pool_type::GetRecycled();
  if (ptr == nullptr) { ptr = new DeviceEvent(device_id, flags); }
  return std::shared_ptr<DeviceEvent>(ptr, &pool_type::Put);
}

// clang-format off
#define FOR_EACH_INT_LT_32(macro)  \
  macro(0)                         \
  macro(1)                         \
  macro(2)                         \
  macro(3)                         \
  macro(4)                         \
  macro(5)                         \
  macro(6)                         \
  macro(7)                         \
  macro(8 + 0)                     \
  macro(8 + 1)                     \
  macro(8 + 2)                     \
  macro(8 + 3)                     \
  macro(8 + 4)                     \
  macro(8 + 5)                     \
  macro(8 + 6)                     \
  macro(8 + 7)                     \
  macro(16 + 0)                    \
  macro(16 + 1)                    \
  macro(16 + 2)                    \
  macro(16 + 3)                    \
  macro(16 + 4)                    \
  macro(16 + 5)                    \
  macro(16 + 6)                    \
  macro(16 + 7)                    \
  macro(24 + 0)                    \
  macro(24 + 1)                    \
  macro(24 + 2)                    \
  macro(24 + 3)                    \
  macro(24 + 4)                    \
  macro(24 + 5)                    \
  macro(24 + 6)                    \
  macro(24 + 7)
// clang-format on

std::shared_ptr<DeviceEvent> GetReusedDeviceEvent(int device_id, unsigned int flags) {
  static thread_local std::vector<std::shared_ptr<DeviceEvent> (*)(unsigned int)> device_id2func{
#define GET_OR_NEW_REUSED_DEVICE_EVENT_ENTRY(device_id) &GetReusedOrNewDeviceEvent<device_id>,

      FOR_EACH_INT_LT_32(GET_OR_NEW_REUSED_DEVICE_EVENT_ENTRY)

#undef GET_OR_NEW_REUSED_DEVICE_EVENT_ENTRY
  };
  return device_id2func.at(device_id)(flags);
}

#endif

}  // namespace oneflow
