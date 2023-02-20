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
#ifndef ONEFLOW_CORE_EP_DEVICE_H_
#define ONEFLOW_CORE_EP_DEVICE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/ep/include/event.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ep/include/allocation_options.h"

namespace oneflow {

namespace ep {

constexpr size_t kMaxAlignmentRequirement = 512;

class DeviceManager;

class Device {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Device);
  Device() = default;
  virtual ~Device() = default;

  virtual void SetAsActiveDevice() = 0;

  virtual DeviceType device_type() const = 0;
  virtual size_t device_index() const = 0;
  virtual DeviceManager* device_manager() const = 0;

  virtual Stream* CreateStream() = 0;
  virtual void DestroyStream(Stream* stream) = 0;

  virtual Event* CreateEvent();
  virtual void DestroyEvent(Event* event);
  virtual void CreateEvents(Event** events, size_t count) = 0;
  virtual void DestroyEvents(Event** events, size_t count) = 0;

  virtual Maybe<void> Alloc(const AllocationOptions& options, void** ptr, size_t size) = 0;
  virtual void Free(const AllocationOptions& options, void* ptr) = 0;
  virtual Maybe<void> AllocPinned(const AllocationOptions& options, void** ptr, size_t size) = 0;
  virtual void FreePinned(const AllocationOptions& options, void* ptr) = 0;
  virtual bool IsStreamOrderedMemoryAllocationSupported() const;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_DEVICE_H_
