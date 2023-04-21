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
#ifndef ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_EAGER_CNCL_COMM_MANAGER_H_
#define ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_EAGER_CNCL_COMM_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/eager_ccl_comm_manager.h"

#include "oneflow/cambricon/collective_communication/cncl_util.h"

namespace oneflow {

class EagerCnclCommMgr final : public EagerCclCommMgr {
 public:
  static const std::string kDefaultStreamName;

  OF_DISALLOW_COPY_AND_MOVE(EagerCnclCommMgr);
  ~EagerCnclCommMgr() override;

  cnclComm_t GetCommForDevice(const std::set<std::pair<int64_t, int64_t>>& device_set);
  cnclComm_t GetCommForDeviceAndStreamName(const std::set<std::pair<int64_t, int64_t>>& device_set,
                                           const std::string& stream_name);

  void CreateCommFromPlan(const Plan& plan) override;
  bool IsAsyncLaunchCclLogicalKernel() const override { return async_launch_cncl_logical_kernel_; }
  void SetAsyncLaunchCclLogicalKernel(bool val) override {
    async_launch_cncl_logical_kernel_ = val;
  }

 private:
  friend class EagerCclCommMgrBuilder;
  // NOTE: default async launch cncl logical kernel is true for better performence.
  EagerCnclCommMgr() : EagerCclCommMgr(), async_launch_cncl_logical_kernel_(true) {}

  std::map<std::set<std::pair<int64_t, int64_t>>, HashMap<int64_t, cnclComm_t>>
      device_set2device_id2comm_;
  std::map<std::string, HashMap<int64_t, cnclComm_t>> device7stream2device_id2comm_;
  std::mutex mutex_;
  bool async_launch_cncl_logical_kernel_;
};

class UserKernelUnifiedCnclCommInitRegistry final {
 public:
  struct Trigger {
    explicit Trigger(const std::string& key) {
      UserKernelUnifiedCnclCommInitRegistry::Instance().Register(key);
    }
  };

  static UserKernelUnifiedCnclCommInitRegistry& Instance() {
    static UserKernelUnifiedCnclCommInitRegistry reg;
    return reg;
  }

  OF_DISALLOW_COPY_AND_MOVE(UserKernelUnifiedCnclCommInitRegistry);
  ~UserKernelUnifiedCnclCommInitRegistry() = default;

  void Register(const std::string& key) {
    bool insert_success = reg_set_.insert(key).second;
    if (!insert_success) {
      std::cerr << key << " was already registered in CnclCommRegistry" << std::endl;
      abort();
    }
  }

  bool IsRegistered(const std::string& key) const { return reg_set_.find(key) != reg_set_.end(); }

 private:
  UserKernelUnifiedCnclCommInitRegistry() = default;
  std::set<std::string> reg_set_;
};

static const std::string kSystemOpPrefix = "sys_op_";

}  // namespace oneflow

#define REGISTER_USER_KERNEL_UNIFIED_CNCL_COMM_INIT(op_type_name) \
  static auto OF_PP_CAT(g_cncl_comm_reg_, __COUNTER__) =          \
      ::oneflow::UserKernelUnifiedCnclCommInitRegistry::Trigger(op_type_name)

#define REGISTER_SYSTEM_OP_KERNEL_UNIFIED_CNCL_COMM_INIT(op_type_case)                     \
  static auto OF_PP_CAT(g_cncl_comm_reg_, __COUNTER__) =                                   \
      ::oneflow::UserKernelUnifiedCnclCommInitRegistry::Trigger(::oneflow::kSystemOpPrefix \
                                                                + std::to_string(op_type_case))

#endif  // ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_EAGER_CNCL_COMM_MANAGER_H_
