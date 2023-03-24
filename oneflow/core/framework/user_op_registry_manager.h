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
#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_REGISTRY_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_REGISTRY_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/core/common/registry_error.h"
#include "oneflow/core/common/op_args_reserved_size.h"

namespace oneflow {

namespace user_op {

class UserOpRegistryMgr final {
 private:
  UserOpRegistryMgr() {}

 public:
  UserOpRegistryMgr(UserOpRegistryMgr const&) = delete;
  UserOpRegistryMgr& operator=(UserOpRegistryMgr const&) = delete;
  static UserOpRegistryMgr& Get();

 public:
  OpRegistry CheckAndGetOpRegistry(const std::string& op_type_name);
  Maybe<void> Register(OpRegistryResult result);
  const OpRegistryResult* GetOpRegistryResult(const std::string& op_type_name);

  OpKernelRegistry CheckAndGetOpKernelRegistry(const std::string& op_type_name);
  Maybe<void> Register(OpKernelRegistryResult result);
  Maybe<const OpKernelRegistryResult*> GetOpKernelRegistryResult(const std::string& op_type_name,
                                                                 const KernelRegContext& ctx);
  Maybe<bool> IsOpKernelRegistered(const std::string& op_type_name, const KernelRegContext& ctx);

  const HashMap<std::string, OpRegistryResult>& GetAllOpRegistryResults() {
    return op_reg_result_;
  };

 private:
  HashMap<std::string, OpRegistryResult> op_reg_result_;
  HashMap<std::string, std::vector<OpKernelRegistryResult>> op_kernel_reg_result_;
};

template<typename RegistryT>
struct UserOpRegisterTrigger final {
  UserOpRegisterTrigger(RegistryT& registry) {
    CatchRegistryError([&]() -> Maybe<void> {
      return UserOpRegistryMgr::Get().Register(JUST(registry.Finish()).GetResult());
    });
  }
};

class UserOpHostMemoryInputRegistry final {
 public:
  UserOpHostMemoryInputRegistry(UserOpHostMemoryInputRegistry const&) = delete;
  UserOpHostMemoryInputRegistry& operator=(UserOpHostMemoryInputRegistry const&) = delete;
  ~UserOpHostMemoryInputRegistry() = default;

  static UserOpHostMemoryInputRegistry& Get();

  Maybe<void> SetHostMemoryInput4Op(const std::string& op_type_name, const std::string& arg_name,
                                    int32_t index);
  bool IsHostMemoryInput4Op(const std::string& op_type_name, const std::string& arg_name,
                            int32_t index) const;

  bool HasHostMemoryInput(const std::string& op_type_name) const;

 private:
  UserOpHostMemoryInputRegistry() {}
  HashMap<std::string, small_vector<std::pair<std::string, int32_t>>>
      op_type_name2host_memory_input_args_;
};

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_OP_HOST_MEMORY_INPUT(op_type_name, arg_name, index)                      \
  COMMAND(CHECK_JUST(user_op::UserOpHostMemoryInputRegistry::Get().SetHostMemoryInput4Op( \
      op_type_name, arg_name, index)));

#define REGISTER_USER_OP(name)                                                                \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpRegistry> OF_PP_CAT( \
      g_register_trigger, __COUNTER__) =                                                      \
      ::oneflow::user_op::UserOpRegistryMgr::Get().CheckAndGetOpRegistry(name)

#define REGISTER_CPU_ONLY_USER_OP(name) REGISTER_USER_OP(name).SupportCpuOnly()

#define REGISTER_NO_GRAD_USER_OP(name) REGISTER_USER_OP(name).NoGrad()

#define REGISTER_NO_GRAD_CPU_ONLY_USER_OP(name) REGISTER_NO_GRAD_USER_OP(name).SupportCpuOnly()

#define REGISTER_USER_KERNEL(name)                                                       \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpKernelRegistry> \
      OF_PP_CAT(g_register_trigger, __COUNTER__) =                                       \
          ::oneflow::user_op::UserOpRegistryMgr::Get().CheckAndGetOpKernelRegistry(name)

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_REGISTRY_MANAGER_H_
