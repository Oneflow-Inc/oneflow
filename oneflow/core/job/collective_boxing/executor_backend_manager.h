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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_BACKEND_MANAGER_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_BACKEND_MANAGER_H_

#include "oneflow/core/job/collective_boxing/executor_backend.h"
#include "oneflow/core/common/device_type.h"

namespace oneflow {

namespace boxing {

namespace collective {

class ExecutorBackendMgr {
 public:
  using Creator = std::function<std::unique_ptr<ExecutorBackend>()>;

  ExecutorBackendMgr(ExecutorBackendMgr const&) = delete;
  ExecutorBackendMgr& operator=(ExecutorBackendMgr const&) = delete;
  static ExecutorBackendMgr& Get();

  template<typename Derived>
  void RegisterExecutorBackendType(DeviceType device_type) {
    executor_backend_reg_result_.emplace(device_type, []() -> std::unique_ptr<ExecutorBackend> {
      return std::make_unique<Derived>();
    });
    vaild_executor_device_types_.emplace_back(device_type);
  }

  std::unique_ptr<ExecutorBackend> NewExecutorBackend(DeviceType device_type) const {
    const auto& it = executor_backend_reg_result_.find(device_type);
    CHECK(it != executor_backend_reg_result_.end());
    return it->second();
  }

  const std::vector<DeviceType>& vaild_executor_device_types() const {
    return vaild_executor_device_types_;
  }

 private:
  ExecutorBackendMgr() = default;

  HashMap<DeviceType, Creator> executor_backend_reg_result_;
  std::vector<DeviceType> vaild_executor_device_types_;
};

#define REGISTER_EXECUTOR_BACKEND(device, Derived) \
  COMMAND(ExecutorBackendMgr::Get().RegisterExecutorBackendType<Derived>(device))

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_BACKEND_MANAGER_H_
