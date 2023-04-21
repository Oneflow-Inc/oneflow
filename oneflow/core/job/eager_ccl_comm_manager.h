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
#ifndef ONEFLOW_CORE_JOB_EAGER_CCL_COMM_MANAGER_H_
#define ONEFLOW_CORE_JOB_EAGER_CCL_COMM_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class EagerCclCommMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerCclCommMgr);
  virtual ~EagerCclCommMgr() = default;

  virtual void CreateCommFromPlan(const Plan& plan) = 0;
  virtual bool IsAsyncLaunchCclLogicalKernel() const = 0;
  virtual void SetAsyncLaunchCclLogicalKernel(bool val) = 0;

  template<typename T>
  T* As() {
    return dynamic_cast<T*>(this);
  }

 protected:
  EagerCclCommMgr() = default;
};

class EagerCclCommMgrBuilder {
 public:
  using Creator = std::function<EagerCclCommMgr*()>;

  EagerCclCommMgrBuilder(EagerCclCommMgrBuilder const&) = delete;
  EagerCclCommMgrBuilder& operator=(EagerCclCommMgrBuilder const&) = delete;
  static EagerCclCommMgrBuilder& Get();

  template<typename Derived>
  void RegisterEagerCclCommMgrType(DeviceType device_type) {
    ccl_comm_mgr_reg_result_->emplace(device_type,
                                      []() -> EagerCclCommMgr* { return new Derived; });
    vaild_ccl_comm_mgr_device_types_.emplace_back(device_type);
  }

  EagerCclCommMgr* NewCclCommMgr(DeviceType device_type) const {
    const auto& it = ccl_comm_mgr_reg_result_->find(device_type);
    CHECK(it != ccl_comm_mgr_reg_result_->end());
    return it->second();
  }

  const std::vector<DeviceType>& vaild_ccl_comm_mgr_device_types() const {
    return vaild_ccl_comm_mgr_device_types_;
  }

 private:
  EagerCclCommMgrBuilder() { ccl_comm_mgr_reg_result_.reset(new std::map<DeviceType, Creator>); }

  std::unique_ptr<std::map<DeviceType, Creator>> ccl_comm_mgr_reg_result_;
  std::vector<DeviceType> vaild_ccl_comm_mgr_device_types_;
};

#define REGISTER_CCL_COMM_MGR(device, Derived) \
  COMMAND(EagerCclCommMgrBuilder::Get().RegisterEagerCclCommMgrType<Derived>(device))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_EAGER_CCL_COMM_MANAGER_H_
