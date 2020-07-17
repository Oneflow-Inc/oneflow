#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_

#include "oneflow/core/framework/user_op_builder.h"
#include "oneflow/core/framework/user_op_grad_builder.h"
#include "oneflow/core/framework/user_op_kernel_builder.h"

namespace oneflow {

namespace user_op {

class UserOpMgr final {
 private:
  UserOpMgr() {}

 public:
  UserOpMgr(UserOpMgr const&) = delete;
  UserOpMgr& operator=(UserOpMgr const&) = delete;
  static UserOpMgr& Get();

 public:
  OpBuilder CheckAndGetOpBuilder(const std::string& op_type_name);
  void Register(OpRegistrationResult& result);
  const OpRegistrationResult* GetOpRegistrationResult(const std::string& op_type_name);

  OpGradBuilder CheckAndGetOpGradBuilder(const std::string& op_type_name);
  void Register(OpGradRegistrationResult& result);
  const OpGradRegistrationResult* GetOpGradRegistrationResult(const std::string& op_type_name);

  OpKernelBuilder CheckAndGetOpKernelBuilder(const std::string& op_type_name);
  void Register(OpKernelRegistrationResult& result);
  const OpKernelRegistrationResult* GetOpKernelRegistrationResult(const std::string& op_type_name);

 private:
  HashMap<std::string, OpRegistrationResult> op_reg_result_;
  HashMap<std::string, OpGradRegistrationResult> op_grad_reg_result_;
  HashMap<std::string, OpKernelRegistrationResult> op_kernel_reg_result_;
};

template<typename BuilderT>
struct UserOpRegisterTrigger final {
  UserOpRegisterTrigger(BuilderT& builder) {
    UserOpMgr::Get().Register(builder.Finish().GetResult());
  }
};

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_OP(name)                                                               \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpBuilder> OF_PP_CAT( \
      g_register_trigger, __COUNTER__) =                                                     \
      ::oneflow::user_op::UserOpMgr::Get().CheckAndGetOpBuilder(name)

#define REGISTER_CPU_ONLY_USER_OP(name) REGISTER_USER_OP(name).SupportCpuOnly()

#define REGISTER_USER_OP_GRAD(name)                                                              \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpGradBuilder> OF_PP_CAT( \
      g_register_trigger, __COUNTER__) =                                                         \
      ::oneflow::user_op::UserOpMgr::Get().CheckAndGetOpGradBuilder(name)

#define REGISTER_USER_KERNEL(name)                                                                 \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpKernelBuilder> OF_PP_CAT( \
      g_register_trigger, __COUNTER__) =                                                           \
      ::oneflow::user_op::UserOpMgr::Get().CheckAndGetOpKernelBuilder(name)

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_