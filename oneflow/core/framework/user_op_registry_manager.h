#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_REGISTRY_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_REGISTRY_MANAGER_H_

#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/user_op_grad_registry.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"

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
  void Register(OpRegistryResult& result);
  const OpRegistryResult* GetOpRegistryResult(const std::string& op_type_name);

  OpGradRegistry CheckAndGetOpGradRegistry(const std::string& op_type_name);
  void Register(OpGradRegistryResult& result);
  const OpGradRegistryResult* GetOpGradRegistryResult(const std::string& op_type_name);

  OpKernelRegistry CheckAndGetOpKernelRegistry(const std::string& op_type_name);
  void Register(OpKernelRegistryResult& result);
  const OpKernelRegistryResult* GetOpKernelRegistryResult(const std::string& op_type_name);

 private:
  HashMap<std::string, OpRegistryResult> op_reg_result_;
  HashMap<std::string, OpGradRegistryResult> op_grad_reg_result_;
  HashMap<std::string, std::vector<OpKernelRegistryResult>> op_kernel_reg_result_;
};

template<typename RegistryT>
struct UserOpRegisterTrigger final {
  UserOpRegisterTrigger(RegistryT& registry) {
    UserOpRegistryMgr::Get().Register(registry.Finish().GetResult());
  }
};

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_OP(name)                                                                \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpRegistry> OF_PP_CAT( \
      g_register_trigger, __COUNTER__) =                                                      \
      ::oneflow::user_op::UserOpRegistryMgr::Get().CheckAndGetOpRegistry(name)

#define REGISTER_CPU_ONLY_USER_OP(name) REGISTER_USER_OP(name).SupportCpuOnly()

#define REGISTER_USER_OP_GRAD(name)                                                               \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpGradRegistry> OF_PP_CAT( \
      g_register_trigger, __COUNTER__) =                                                          \
      ::oneflow::user_op::UserOpRegistryMgr::Get().CheckAndGetOpGradRegistry(name)

#define REGISTER_USER_KERNEL(name)                                                       \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpKernelRegistry> \
      OF_PP_CAT(g_register_trigger, __COUNTER__) =                                       \
          ::oneflow::user_op::UserOpRegistryMgr::Get().CheckAndGetOpKernelRegistry(name)

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_REGISTRY_MANAGER_H_