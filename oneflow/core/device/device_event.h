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

#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/single_thread_obj_pool.h"

namespace oneflow {

class DeviceEvent final {
 public:
  DeviceEvent(const DeviceEvent&) = delete;
  DeviceEvent(DeviceEvent&&) = delete;

  DeviceEvent(int device_id, unsigned int flags);
  ~DeviceEvent();

  int device_id() const { return device_id_; }
  bool Query() const;

  cudaEvent_t* mut_event() { return &event_; }

 private:
  int device_id_;
  cudaEvent_t event_;
};

class DeviceEventProvider {
 public:
  DeviceEventProvider(const DeviceEventProvider&) = delete;
  DeviceEventProvider(DeviceEventProvider&&) = delete;
  explicit DeviceEventProvider(int device_id) : device_events_(), device_id_(device_id) {}
  virtual ~DeviceEventProvider() = default;

  std::shared_ptr<DeviceEvent> GetSingleThreadReusedDeviceEventWithFlags(unsigned int flags) {
    return device_events_.make_shared(device_id_, flags);
  }

 private:
  obj_pool::SingleThreadObjPool<DeviceEvent, obj_pool::kDisableReconstruct> device_events_;
  int device_id_;
};

class QueryEventProvider : public DeviceEventProvider {
 public:
  QueryEventProvider(const QueryEventProvider&) = delete;
  QueryEventProvider(QueryEventProvider&&) = delete;
  using DeviceEventProvider::DeviceEventProvider;
  virtual ~QueryEventProvider() = default;

  std::shared_ptr<DeviceEvent> GetSingleThreadReusedDeviceEvent() {
    return GetSingleThreadReusedDeviceEventWithFlags(cudaEventBlockingSync
                                                     | cudaEventDisableTiming);
  }
};

}  // namespace oneflow

#endif

#endif  // ONEFLOW_CORE_DEVICE_DEVICE_EVENT_H_
