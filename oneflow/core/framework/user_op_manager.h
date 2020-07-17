#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_

#include "oneflow/core/framework/op_builder.h"

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

 private:
  HashMap<std::string, OpRegistrationResult> op_reg_result_;
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

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_