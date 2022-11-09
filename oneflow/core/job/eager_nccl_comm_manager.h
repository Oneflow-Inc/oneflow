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
#ifndef ONEFLOW_CORE_JOB_EAGER_NCCL_COMM_MANAGER_H_
#define ONEFLOW_CORE_JOB_EAGER_NCCL_COMM_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class EagerNcclCommMgr final {
 public:
  static const std::string kDefaultStreamName;

  OF_DISALLOW_COPY_AND_MOVE(EagerNcclCommMgr);
  ~EagerNcclCommMgr();

  ncclComm_t GetCommForDevice(const std::set<std::pair<int64_t, int64_t>>& device_set);
  ncclComm_t GetCommForDeviceAndStreamName(const std::set<std::pair<int64_t, int64_t>>& device_set,
                                           const std::string& stream_name);

  void CreateCommFromPlan(const Plan& plan);
  bool IsAsyncLaunchNcclLogicalKernel() const { return async_launch_nccl_logical_kernel_; }
  void SetAsyncLaunchNcclLogicalKernel(bool val) { async_launch_nccl_logical_kernel_ = val; }

 private:
  friend class Singleton<EagerNcclCommMgr>;
  // NOTE(chengcheng): default async launch nccl logical kernel is true for better performence.
  EagerNcclCommMgr() : async_launch_nccl_logical_kernel_(true) {}

  std::map<std::set<std::pair<int64_t, int64_t>>, HashMap<int64_t, ncclComm_t>>
      device_set2device_id2comm_;
  std::map<std::string, HashMap<int64_t, ncclComm_t>> device7stream2device_id2comm_;
  std::mutex mutex_;
  bool async_launch_nccl_logical_kernel_;
};

class UserKernelUnifiedNcclCommInitRegistry final {
 public:
  struct Trigger {
    explicit Trigger(const std::string& key) {
      UserKernelUnifiedNcclCommInitRegistry::Instance().Register(key);
    }
  };

  static UserKernelUnifiedNcclCommInitRegistry& Instance() {
    static UserKernelUnifiedNcclCommInitRegistry reg;
    return reg;
  }

  OF_DISALLOW_COPY_AND_MOVE(UserKernelUnifiedNcclCommInitRegistry);
  ~UserKernelUnifiedNcclCommInitRegistry() = default;

  void Register(const std::string& key) {
    bool insert_success = reg_set_.insert(key).second;
    if (!insert_success) {
      std::cerr << key << " was already registered in NcclCommRegistry" << std::endl;
      abort();
    }
  }

  bool IsRegistered(const std::string& key) const { return reg_set_.find(key) != reg_set_.end(); }

 private:
  UserKernelUnifiedNcclCommInitRegistry() = default;
  std::set<std::string> reg_set_;
};

static const std::string kSystemOpPrefix = "sys_op_";

}  // namespace oneflow

#define REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT(op_type_name) \
  static auto OF_PP_CAT(g_nccl_comm_reg_, __COUNTER__) =          \
      ::oneflow::UserKernelUnifiedNcclCommInitRegistry::Trigger(op_type_name)

#define REGISTER_SYSTEM_OP_KERNEL_UNIFIED_NCCL_COMM_INIT(op_type_case)                     \
  static auto OF_PP_CAT(g_nccl_comm_reg_, __COUNTER__) =                                   \
      ::oneflow::UserKernelUnifiedNcclCommInitRegistry::Trigger(::oneflow::kSystemOpPrefix \
                                                                + std::to_string(op_type_case))

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_JOB_EAGER_NCCL_COMM_MANAGER_H_
