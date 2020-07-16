#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_

#include "oneflow/core/framework/op_registration.h"

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
  void Register(OpBuilder& builder);
};

template<typename BuilderT>
struct UserOpRegisterTrigger final {
  UserOpRegisterTrigger(BuilderT& builder) { UserOpManager::Get().Register(builder.Finish()); }
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_MANAGER_H_