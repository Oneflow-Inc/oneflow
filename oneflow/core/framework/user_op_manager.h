#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_

#include "oneflow/core/framework/op_builder.h"

namespace oneflow {

namespace user_op {

class UserOpManager final {
 private:
  UserOpManager() {}

 public:
  UserOpManager(UserOpManager const&) = delete;
  UserOpManager& operator=(UserOpManager const&) = delete;

 public:
  static UserOpManager& Get();

  OpBuilder GetOpBuilder() { return OpBuilder(); }
  void Register(OpRegistResult& result);
  const OpBuildResult* GetOpInfo(const std::string& op_type_name);

 private:
  HashMap<std::string, OpBuildResult> op_info_;
};

template<typename BuilderT>
struct UserOpRegisterTrigger final {
  UserOpRegisterTrigger(BuilderT& builder) {
    UserOpManager::Get().Register(builder.Finish().GetResult());
  }
};

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_OP(name)                                                               \
  static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpBuilder> OF_PP_CAT( \
      g_registrar, __COUNTER__) =                                                            \
      ::oneflow::user_op::UserOpManager::Get().GetOpBuilder().Name(name)

#define REGISTER_CPU_ONLY_USER_OP(name) REGISTER_USER_OP(name).SupportCpuOnly()

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_