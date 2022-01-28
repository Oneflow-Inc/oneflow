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
#ifndef ONEFLOW_CORE_EP_CPU_CPU_DEVICE_H_
#define ONEFLOW_CORE_EP_CPU_CPU_DEVICE_H_

#include "oneflow/core/ep/include/device.h"

namespace oneflow {

namespace ep {

class CpuDevice : public Device {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDevice);
  explicit CpuDevice(DeviceManager* device_manager)
      : device_manager_(device_manager), num_threads_(1) {}
  ~CpuDevice() override = default;

  void SetAsActiveDevice() override;
  void SetNumThreads(size_t num_threads) { num_threads_ = num_threads; }
  size_t GetNumThreads() { return num_threads_; }

  DeviceType device_type() const override { return DeviceType::kCPU; }
  size_t device_index() const override { return 0; }
  DeviceManager* device_manager() const override { return device_manager_; }

  Stream* CreateStream() override;
  void DestroyStream(Stream* stream) override;

  void CreateEvents(Event** events, size_t count) override;
  void DestroyEvents(Event** events, size_t count) override;

  Maybe<void> Alloc(const AllocationOptions& options, void** ptr, size_t size) override;
  void Free(const AllocationOptions& options, void* ptr) override;
  Maybe<void> AllocPinned(const AllocationOptions& options, void** ptr, size_t size) override;
  void FreePinned(const AllocationOptions& options, void* ptr) override;

 private:
  DeviceManager* device_manager_;
  size_t num_threads_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_CPU_CPU_DEVICE_H_
