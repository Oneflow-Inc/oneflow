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
#ifndef ONEFLOW_CORE_DEVICE_DEVICE_EVENT_H_
#define ONEFLOW_CORE_DEVICE_DEVICE_EVENT_H_

#include "oneflow/core/common/obj_pool.h"

#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class DeviceEvent final {
 public:
  DeviceEvent(const DeviceEvent&) = delete;
  DeviceEvent(DeviceEvent&&) = delete;

  DeviceEvent(int device_id, unsigned int flags);
  ~DeviceEvent();

  bool Query() const;

  cudaEvent_t* mut_event() { return &event_; }

 private:
  int device_id_;
  cudaEvent_t event_;
};

std::shared_ptr<DeviceEvent> GetReusedDeviceEvent(int device_id, unsigned int flags);

}  // namespace oneflow

#endif

#endif  // ONEFLOW_CORE_DEVICE_DEVICE_EVENT_H_
