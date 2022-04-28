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
#ifndef ONEFLOW_CORE_EP_NPU_NPU_DEVICE_MANAGER_H_
#define ONEFLOW_CORE_EP_NPU_NPU_DEVICE_MANAGER_H_
#ifdef WITH_NPU
#include "oneflow/core/ep/include/device_manager.h"
#include "acl/acl.h"
#include "acl/acl_base.h"

namespace oneflow {

namespace ep {

class NpuDeviceManager : public DeviceManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NpuDeviceManager);
  explicit NpuDeviceManager(DeviceManagerRegistry* registry);
  ~NpuDeviceManager() override;

  DeviceManagerRegistry* registry() const override;
  std::shared_ptr<Device> GetDevice(size_t device_index) override;
  size_t GetDeviceCount(size_t primary_device_index) override;
  size_t GetDeviceCount() override;
  size_t GetActiveDeviceIndex() override;
  void SetActiveDeviceByIndex(size_t device_index) override;
  void SetDeviceNumThreads(size_t num_threads);

 private:
  size_t device_num_threads_;
  std::mutex devices_mutex_;
  std::vector<std::shared_ptr<Device>> devices_;
  DeviceManagerRegistry* registry_;
};

}  // namespace ep

}  // namespace oneflow

#endif // WITH_NPU

#endif  // ONEFLOW_CORE_EP_NPU_NPU_DEVICE_MANAGER_H_
